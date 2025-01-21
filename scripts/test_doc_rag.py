from datasets import load_dataset
from dotenv import load_dotenv
from src.vidore_benchmark.evaluation.evaluate import evaluate_dataset
from src.vidore_benchmark.retrievers.jina_clip_retriever import JinaClipRetriever
from src.vidore_benchmark.retrievers.colpali_retriever import ColPaliRetriever
import json
import os
from datetime import datetime

load_dotenv(override=True)

def main():
    """
    Example script for a Python usage of the Vidore Benchmark.
    """
    # == Local
    # my_retriever = ColPaliRetriever('../models/vidore/colpali-v1.2')
    
    # == Download
    # my_retriever = JinaClipRetriever("jinaai/jina-clip-v1")
    my_retriever = ColPaliRetriever("vidore/colpali-v1.2")
    
    dataset = load_dataset("vidore/syntheticDocQA_artificial_intelligence_test", split="test")
    
    batch_size = 2
    metrics = evaluate_dataset(my_retriever, dataset, batch_query=batch_size, batch_passage=batch_size, batch_score=batch_size)
    
    print(metrics)
    
    # Create eval_res directory if it doesn't exist
    os.makedirs('eval_res', exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'eval_res/colpali_eval_{timestamp}.json'
    
    # Save metrics to JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=batch_size)
    
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    main()