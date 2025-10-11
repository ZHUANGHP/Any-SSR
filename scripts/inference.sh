#!bin/bash
BASE_PATH="/home/yanyue/TRACE/output_models"
port=$(shuf -i25000-30000 -n1)
deepspeed --include=localhost:6 --master_port $port inference/infer_almoe_total_debug.py \
   --router_weight_path /home/yanyue/TRACE/9_18test \
   --data_path /home/yanyue/LLM-CL-Benchmark_5000/ \
   --inference_tasks NumGLUE-cm,NumGLUE-ds,FOMC,20Minuten,C-STANCE,Py150,MeetingBank \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --inference_model_path "$BASE_PATH/NumGLUE-cm/0","$BASE_PATH/NumGLUE-ds/0","$BASE_PATH/FOMC/0","$BASE_PATH/20Minuten/0","$BASE_PATH/C-STANCE/0","$BASE_PATH/Py150/0","$BASE_PATH/MeetingBank/0" \
   --max_prompt_len 1024 \
   --max_ans_len 512 \
   --seed 1234 \
   --deepspeed \
   --inference_output_path /home/yanyue/TRACE/inference_result/test_9_17

