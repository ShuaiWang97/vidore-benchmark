export CUDA_VISIBLE_DEVICES=0  # Only use GPU 0

# Add both the current directory and src directory to PYTHONPATH
export PYTHONPATH=$(pwd):$(pwd)/src:$PYTHONPATH

# python scripts/test_doc_rag.py --model "SigLIP" --dataset "vidore/syntheticDocQA_artificial_intelligence_test_tesseract"
 
# python3 scripts/test_doc_rag.py --model "SigLIP" --dataset "vidore/syntheticDocQA_artificial_intelligence_test"


# python3 scripts/test_doc_rag.py --model "colpali" --dataset "vidore/syntheticDocQA_artificial_intelligence_test_ocr_chunk"  --document_input text




# Define the document input types
INPUT_TYPES=("text" "image" "image+text")
MODEL_TYPES=("SigLIP" "ColPali")
DATASETS=(
          "vidore/docvqa_test_subsampled_ocr_chunk" \
          "vidore/arxivqa_test_subsampled_ocr_chunk" \
            "vidore/infovqa_test_subsampled_ocr_chunk" \
            "vidore/tabfquad_test_subsampled_ocr_chunk" \
            "vidore/tatdqa_test_ocr_chunk" \
            "vidore/shiftproject_test_ocr_chunk"\
            "vidore/syntheticDocQA_artificial_intelligence_test_ocr_chunk"\
            "vidore/syntheticDocQA_energy_test_ocr_chunk" \
            "vidore/syntheticDocQA_government_reports_test_ocr_chunk"\
            "vidore/syntheticDocQA_healthcare_industry_test_ocr_chunk"\
            )


# Loop through each model and input type and run the Python script
for model_type in "${MODEL_TYPES[@]}"; do
    for input_type in "${INPUT_TYPES[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            echo "Running for model: $model_type and document input type: $input_type"
            sbatch snellius_job.sh  $model_type $input_type $dataset
        done
    done
done




# #for model_type in "mtnp0923" "mtnp0925" "mtnp1114"
# for model_type in "SigLIP" "colpali"
#     for input_type in "text" "image" "image+text"
#     do
#         echo "Hello, Welcome for model_type $model_type."
#         sbatch snellius_job.sh $model_type $input_type
#     done