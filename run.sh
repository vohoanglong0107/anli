python main.py \
  --model_class_name "MiniLM-L6" \
  -n 1 \
  -g 1 \
  --single_gpu \
  -nr 0 \
  --max_length 156 \
  --gradient_accumulation_steps 1 \
  --per_gpu_train_batch_size 256 \
  --per_gpu_eval_batch_size 512 \
  --save_prediction \
  --train_data \
snli_train:none,mnli_train:none,fever_train:none,anli_r1_train:none,anli_r2_train:none,anli_r3_train:none \
  --train_weights \
1,1,1,10,20,10 \
  --eval_data \
snli_train:none,mnli_train:none,fever_train:none,anli_r1_train:none,anli_r2_train:none,anli_r3_train:none \
  --eval_frequency 4000 \
  --experiment_name "MiniLM-L6|snli+mnli+fnli+r1*10+r2*20+r3*10|nli" \
  --save_pretrained \
  --push_to_hub

# uncomment to test warmup
# python main.py \
#   --model_class_name "MiniLM-L6" \
#   -n 1 \
#   -g 1 \
#   --single_gpu \
#   -nr 0 \
#   --max_length 156 \
#   --gradient_accumulation_steps 1 \
#   --per_gpu_train_batch_size 128 \
#   --per_gpu_eval_batch_size 256 \
#   --save_prediction \
#   --train_data \
# anli_r1_train:none \
#   --train_weights \
# 1 \
#   --epochs 1 \
#   --eval_data \
# anli_r1_train:none \
#   --eval_frequency 30 \
#   --experiment_name "MiniLM-L6|snli+mnli+fnli+r1*10+r2*20+r3*10|nli" \
#   --save_pretrained \
#   --push_to_hub