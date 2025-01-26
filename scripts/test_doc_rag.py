from datasets import load_dataset
from dotenv import load_dotenv
from src.vidore_benchmark.evaluation.evaluate import evaluate_dataset
from src.vidore_benchmark.retrievers.jina_clip_retriever import JinaClipRetriever
from src.vidore_benchmark.retrievers.colpali_retriever import ColPaliRetriever
import json
import os
from datetime import datetime

load_dotenv(override=True)

def load_model():
    # == Local
    # my_retriever = ColPaliRetriever('../models/vidore/colpali-v1.2')

    # == Download
    # my_retriever = JinaClipRetriever("jinaai/jina-clip-v1")
    # my_retriever = ColPaliRetriever("vidore/colpali-v1.2")
    return my_retriever

def main():
    """
    Example script for a Python usage of the Vidore Benchmark.
    """

    dataset = load_dataset("vidore/syntheticDocQA_artificial_intelligence_test", split="test")
    
    # Process dataset in groups of 10
    start_idx = 50
    end_idx = 60
    step = 10
    
    # Create eval_res directory if it doesn't exist
    os.makedirs('eval_res', exist_ok=True)
    
    all_metrics = []
    my_retriever = load_model()
    
    for i in range(start_idx, end_idx, step):
        group_dataset = dataset.select(range(i, min(i + step, end_idx)))
        print(f"\nProcessing group {i}-{min(i + step, end_idx)}")
        
        batch_size = 2
        metrics = evaluate_dataset(my_retriever, ds=group_dataset, 
                                 batch_query=batch_size, batch_passage=batch_size, 
                                 batch_score=batch_size, rerank=True)
        
        print(f"Group metrics: {metrics}")
        all_metrics.append(metrics)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'eval_res/colpali_eval_{timestamp}.json'
    
    # Save all metrics to JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            'group_metrics': all_metrics,
            'groups': [{'start': i, 'end': min(i + step, end_idx)} 
                      for i in range(start_idx, end_idx, step)]
        }, f, indent=2)
    
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    main()