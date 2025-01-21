export CUDA_VISIBLE_DEVICES=0  # Only use GPU 0

# Add both the current directory and src directory to PYTHONPATH
export PYTHONPATH=$(pwd):$(pwd)/src:$PYTHONPATH

python3 scripts/test_doc_rag.py