ATT_METHOD=word # word, structure
MAX_PER=5
MODEL_PATH=../QASlow/results/dialogpt # bart, t5, dialogpt
DATASET=blended_skill_talk # blended_skill_talk, conv_ai_2, empathetic_dialogues, AlekseyKorshuk/persona-chat
FITNESS=length # performance, length, combined
NUM_SAMPLES=50
MAX_LENGTH=150
SELECT_BEAMS=2
RES_DIR=results # where to save logs and results


CUDA_VISIBLE_DEVICES=0 python -W ignore DGattack.py \
    --attack_strategy $ATT_METHOD \
    --max_per $MAX_PER \
    --model_name_or_path $MODEL_PATH \
    --dataset $DATASET \
    --max_num_samples $NUM_SAMPLES \
    --max_len $MAX_LENGTH \
    --select_beams $SELECT_BEAMS \
    --out_dir $RES_DIR \
    --fitness $FITNESS \