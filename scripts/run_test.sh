export CUDA_VISIBLE_DEVICES=1
export HF_TOKEN=your_hf_token_here
source ~/anaconda3/etc/profile.d/conda.sh
# export TARGET_PERSON="n000063"
export TARGET_PERSON="n000142"
conda activate sam

python run_edit.py \
  --input_dir="./data/VGGFace2/$TARGET_PERSON/set_A/" \
  --output_dir="./data/VGGFace2/$TARGET_PERSON/set_A_edited/" \
  --save_masks \
  --nonce=89

conda activate anti-dreambooth
python run_pgd.py \
  --clean_data_dir="./data/VGGFace2/$TARGET_PERSON/set_A/" \
  --trans_data_dir="./data/VGGFace2/$TARGET_PERSON/set_A_edited/" \
  --save_dir="./data/VGGFace2/$TARGET_PERSON/set_A_cloaked/"

# run with clean images
export EXPERIMENT_NAME="test"
export MODEL_PATH="Manojb/stable-diffusion-2-1-base"
export CLASS_DIR="data/class-person"
export CLEAN_TRAIN_DIR="data/VGGFace2/$TARGET_PERSON/set_A"
export REF_MODEL_PATH="outputs/$EXPERIMENT_NAME/${TARGET_PERSON}_CLEAN"
accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$CLEAN_TRAIN_DIR\
  --class_data_dir=$CLASS_DIR \
  --output_dir=$REF_MODEL_PATH \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=16 \
  --save_model=1

# run with cloaked images
export EXPERIMENT_NAME="test"
export MODEL_PATH="Manojb/stable-diffusion-2-1-base"
export CLASS_DIR="data/class-person"
export CLEAN_TRAIN_DIR="data/VGGFace2/$TARGET_PERSON/set_A_cloaked"
export REF_MODEL_PATH="outputs/$EXPERIMENT_NAME/${TARGET_PERSON}_CLOAKED"
accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$CLEAN_TRAIN_DIR\
  --class_data_dir=$CLASS_DIR \
  --output_dir=$REF_MODEL_PATH \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=16 \
  --save_model=1

python run_detect.py \
  --clean_path="./outputs/test/${TARGET_PERSON}_CLEAN/" \
  --cloak_path="./outputs/test/${TARGET_PERSON}_CLOAKED/"