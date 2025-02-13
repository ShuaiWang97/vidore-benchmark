from datasets import load_dataset
import argparse
from dotenv import load_dotenv
from src.vidore_benchmark.evaluation.evaluate import evaluate_dataset
from src.vidore_benchmark.retrievers.jina_clip_retriever import JinaClipRetriever
from src.vidore_benchmark.retrievers.colpali_retriever import ColPaliRetriever
from src.vidore_benchmark.retrievers.siglip_retriever import SigLIPRetriever
from src.vidore_benchmark.retrievers.biqwen2_retriever import BiQwen2Retriever
import json
import os
from datetime import datetime

load_dotenv(override=True)

def load_model(model_name):
    # == Local
    # my_retriever = ColPaliRetriever('../models/vidore/colpali-v1.2')

    # == Download
    if model_name == "jina-clip":
        my_retriever = JinaClipRetriever()
    elif model_name == "SigLIP":
        my_retriever = SigLIPRetriever()
    elif model_name == "ColPali":
        my_retriever = ColPaliRetriever("vidore/colpali-v1.2")
    elif model_name == "BiQwen2":
        my_retriever = BiQwen2Retriever("vidore/colqwen2-v1.0")
    return my_retriever

def main(args):
    """
    Example script for a Python usage of the Vidore Benchmark.
    """

    dataset_name = args.dataset
    model_name = args.model
    document_input = args.document_input
    size = args.size
    # dataset = load_dataset("vidore/syntheticDocQA_artificial_intelligence_test", split="test")
    dataset = load_dataset(dataset_name, split="test")
    dataset = dataset.filter(lambda x: x['chunk_type'] == 'text')
    # check the average number of words  of text in the dataset
    print("average number of words in the dataset", sum([len(x['text_description'].split()) for x in dataset])/len(dataset))
    # Process dataset in groups of 10
    start_idx = 0
    end_idx = int(len(dataset)*size)
    step = end_idx
    
    # Create eval_res directory if it doesn't exist
    os.makedirs('eval_res', exist_ok=True)
    
    all_metrics = []
    my_retriever = load_model(model_name)
    
    for i in range(start_idx, end_idx, step):
        group_dataset = dataset.select(range(i, min(i + step, end_idx)))
        print(f"\nProcessing group {i}-{min(i + step, end_idx)}")
        
        batch_size = 10
        metrics = evaluate_dataset(my_retriever, ds=group_dataset, 
                                 batch_query=batch_size, batch_passage=batch_size, 
                                 batch_score=batch_size, rerank=False,document_input=document_input)
        
        print(f"Group metrics: {metrics}")
        all_metrics.append(metrics)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if not os.path.exists('eval_res'):
        os.makedirs('eval_res')
    if "syntheticDocQA" in dataset_name:
        name = dataset_name.split('/syntheticDocQA_')[-1][:5]
    else:
        name = dataset_name.split('/')[-1][:5]
    filename = f'eval_res/colpali_eval_{name}_{model_name}_{document_input}_{timestamp}.json'
    
    # Save all metrics to JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            'group_metrics': all_metrics,
            'groups': [{'start': i, 'end': min(i + step, end_idx)} 
                      for i in range(start_idx, end_idx, step)],
            'args': vars(args),
        }, f, indent=2)
    print(f"args: {args}")
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vidore Benchmark Evaluation Script")
    parser.add_argument("--dataset", type=str,
                        default = "vidore/syntheticDocQA_artificial_intelligence_test_ocr_chunk",help="Name of the dataset to load")
    parser.add_argument("--model", type=str, default = "ColPali", 
                        choices=["jina-clip", "ColPali", "BiQwen2","SigLIP"], help="Name of the model to use")
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction,  help="Whether to rerank the results")

    parser.add_argument("--document_input", type=str, default="image", choices=["image", "text","image+text"],
                        help="Input type for the document")
   
    parser.add_argument("--size", type=float, default=1, help="Size of the dataset")
    args = parser.parse_args()

    print(f"Running evaluation with args: {args}")
    
    main(args)