#!/bin/bash

# Per captioning interattivo conviene usare un checkpoint "mix", ad esempio:
# MODEL_PATH="HuggingFaceModel/paligemma-3b-mix-224"
MODEL_PATH="HuggingFaceModel/paligemma-3b-pt-224"
PROMPT="caption en "
IMAGE_FILE_PATH="test_images/image2.jpg"
MAX_TOKENS_TO_GENERATE=100
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="False"

python3 inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \
