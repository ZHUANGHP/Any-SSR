#!bin/bash
port=$(shuf -i25000-30000 -n1)
deepspeed --include=localhost:5 --master_port $port training/main_anamoe.py \
   --data_path /home/yanyue/LLM-CL-Benchmark_5000/ \
   --dataset_name 20Minuten \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 16 \
   --max_prompt_len 1024 \
   --max_ans_len 512 \
   --learning_rate 1e-4 \
   --weight_decay 0. \
   --num_train_epochs 10 \
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 2 \
   --deepspeed \
   --print_loss \
   --CL_method anamoe \
   --output_dir /home/yanyue/TRACE/outputs_models/20Minuten
