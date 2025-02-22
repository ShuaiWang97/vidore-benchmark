from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from tqdm import tqdm

from vidore_benchmark.compression.token_pooling import BaseEmbeddingPooler
from vidore_benchmark.retrievers.bm25_retriever import BM25Retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.retrievers.jina_clip_retriever import JinaClipRetriever
from vidore_benchmark.retrievers.colpali_retriever import ColPaliRetriever
from vidore_benchmark.retrievers.siglip_retriever import SigLIPRetriever

from vidore_benchmark.utils.iter_utils import batched
from vidore_benchmark.utils.rag import chunking, merge_retrieval_results
from vidore_benchmark.models.run_qw_7B import QwenVLModel
from vidore_benchmark.models.run_openai import OpenAIImageRanker


def get_rerank_docs_results(doc_results, ds, k=8):
    """
    Rerank top k results using vision language model
    Args:
        doc_results: Original retrieval results
        ds: Dataset containing images
        vision_retriever: Vision retriever instance with VLM
        k: Number of top results to rerank
    """
    # Initialize QwenVL model
    # rerank_model = QwenVLModel()
    rerank_model = OpenAIImageRanker()

    reranked_results = {}
    
    for query_id, doc_scores in doc_results.items():
        # Get top k doc_ids and scores
        top_k_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Get corresponding images - using original doc_ids without parsing
        images = [ds[idx]["image"] for idx, _ in enumerate(top_k_items)]
        query = query_id  # Assuming query_id is the actual query text
        
        # Rerank using VLM
        rerank_scores = rerank_model.rerank_with_vlm(query, images)
        
        print("==rerank_scores==", rerank_scores)
        # import pdb; pdb.set_trace()

        # Create new results dict with reranked scores, keeping original doc_ids
        reranked_doc_scores = {
            doc_id: float(score)  
            for (doc_id, _), score in zip(top_k_items, rerank_scores)
        }
        reranked_results[query_id] = reranked_doc_scores
    
    return reranked_results

def fuse_passage_tensors(tensors1, tensors2, fuse_method):
    # Ensure both lists have the same length
    assert len(tensors1) == len(tensors1), "Tensor lists must have the same length"
    
    # import pdb; pdb.set_trace()
    fused_tensors = []
    for tensor1, tensor2 in zip(tensors1, tensors2):

        if fuse_method == "average":
            fused_tensor = (tensor1 + tensor2) / 2
        elif fuse_method == 'concat':
            fused_tensor = torch.cat([tensor1, tensor2], dim=0)
        elif fuse_method == 'Multiplication':
            fused_tensor = tensor1 * tensor2
            fused_tensor = fused_tensor / fused_tensor.norm(dim=-1, keepdim=True)

        fused_tensors.append(fused_tensor)
    
    return fused_tensors

def evaluate_dataset(
    vision_retriever: VisionRetriever,
    ds: Dataset,
    batch_query: int,
    batch_passage: int,
    batch_score: Optional[int] = None,
    embedding_pooler: Optional[BaseEmbeddingPooler] = None,
    rerank = True,
    document_input = str,
    chunk_size = 100,
) -> Dict[str, Optional[float]]:
    """
    Evaluate the model on a given dataset using the MTEB metrics.

    NOTE: The dataset should contain the following columns:
    - query: the query text
    - image_filename: the filename of the image
    - image: the image (PIL.Image) if `use_visual_embedding` is True
    - text_description: the text description (i.e. the page caption or the text chunks) if
        `use_visual_embedding` is False
    """
    # Dataset: sanity check
    passage_column_name = "image" if vision_retriever.use_visual_embedding else "text_description"
    required_columns = ["query", passage_column_name, "image_filename"]

    if not all(col in ds.column_names for col in required_columns):
        raise ValueError(f"Dataset should contain the following columns: {required_columns}")

    # Remove `None` queries (i.e. pages for which no question was generated) and duplicates
    # queries = list(set(ds["query"]))
    # --> old buggy behavior - this differs from colpali-engine implementation where duplicates are NOT removed
    # for fairness with externally evaluated retrievers since bug, we maintain this behavior and remove duplicates
    # This slightly boosts scores on docvqa typically
    seen_queries = set()
    queries = []
    for query in ds["query"]:
        if query is not None and query not in seen_queries:
            queries.append(query)
            seen_queries.add(query)

    if len(queries) == 0:
        raise ValueError("All queries are None")

    # Edge case: using the BM25Retriever
    if isinstance(vision_retriever, BM25Retriever):
        passages = ds[passage_column_name]
        scores = vision_retriever.get_scores_bm25(queries = queries, passages = passages)
        relevant_docs, results = vision_retriever.get_relevant_docs_results(ds, queries, scores)
        metrics = vision_retriever.compute_metrics(relevant_docs, results)
        return metrics

    # Get the embeddings for the queries and passages
    emb_queries = vision_retriever.forward_queries(queries, batch_size = batch_query)

    # NOTE: To prevent overloading the RAM for large datasets, we will load the passages (images)
    # that will be fed to the model in batches (this should be fine for queries as their memory footprint
    # is negligible. This optimization is about efficient data loading, and is not related to the model's
    # forward pass which is also batched.
    image_emb_passages: List[torch.Tensor] = []

    dataloader_prebatch_size = 10 * batch_passage

    for passage_batch in tqdm(
        batched(ds, n=dataloader_prebatch_size),
        desc="Dataloader pre-batching",
        total=math.ceil(len(ds) / (dataloader_prebatch_size)),
    ):
        passages: List[Any] = [db[passage_column_name] for db in passage_batch]
        batch_emb_passages = vision_retriever.forward_passages(passages, batch_size=batch_passage)
        if isinstance(batch_emb_passages, torch.Tensor):
            batch_emb_passages = list(torch.unbind(batch_emb_passages))
            image_emb_passages.extend(batch_emb_passages)
        else:
            image_emb_passages.extend(batch_emb_passages)

    text_chunks = []
    chunk_to_doc_mapping = []  # To track which chunk belongs to which document

    for idx, doc in enumerate(ds):
        # text = get_docu(ds)
        doc_text = doc["text_description"]
        doc_chunks = chunking(doc_text, chunk_size)
        text_chunks.extend(doc_chunks)
        chunk_to_doc_mapping.extend([idx] * len(doc_chunks))
 
    # import pdb; pdb.set_trace()
    # Process text chunks
    text_emb_passages: List[torch.Tensor] = []
    for chunk_batch in tqdm(
        batched(text_chunks, n=batch_passage),
        desc="Processing text passages",
        total=math.ceil(len(text_chunks) / batch_passage),
    ):
        batch_emb_passages = vision_retriever.forward_queries(chunk_batch, batch_size=batch_passage)
        if isinstance(batch_emb_passages, torch.Tensor):
            batch_emb_passages = list(torch.unbind(batch_emb_passages))
        text_emb_passages.extend(batch_emb_passages)

    # import pdb; pdb.set_trace()
    # Get separate similarity scores for image and text
    image_scores = vision_retriever.get_scores(emb_queries, image_emb_passages, batch_size=batch_score)
    text_scores = vision_retriever.get_scores(emb_queries, text_emb_passages, batch_size=batch_score)

    # Get relevant documents and results separately
    _, image_results = vision_retriever.get_relevant_docs_results(
        ds, queries, image_scores
    )

    # For text results, we need to map chunk results back to original documents
    relevant_docs, text_results = vision_retriever.get_relevant_docs_results(
        ds, queries, text_scores, chunk_mapping = chunk_to_doc_mapping
    )

    if document_input == "image":
        results = image_results
    elif document_input == "text":
        results = text_results
    elif document_input == "image+text":
        results = merge_retrieval_results(
            image_results, 
            text_results, 
            merge_strategy='weighted_sum',
            image_weight=0.5,  # Adjust weights based on your specific use case
            text_weight=0.5
        )

    if rerank:
        rerank_results = get_rerank_docs_results(results, ds)
        metrics = vision_retriever.compute_metrics(relevant_docs, rerank_results)
        metric_origin = vision_retriever.compute_metrics(relevant_docs, results)
        
        print(f"===1\n{metrics}")
        print(f"===2\n{metric_origin}")
    else:
        metrics = vision_retriever.compute_metrics(relevant_docs, results)

    return metrics
