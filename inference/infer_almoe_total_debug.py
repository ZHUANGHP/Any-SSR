"""
    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
"""


# import sys 
# if hasattr(sys, 'gettrace') and sys.gettrace(): 
#     print("⚠️ 检测到已有调试器附加，强制退出！")
#     sys.exit(1) 

# !/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import os
import math
import sys
from tqdm import tqdm
import pandas as pd
import debugpy

print('-----------------------------------------------------------------------')

try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9797))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import deepspeed
import json

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_collator import DataCollator
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, \
    get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.model.model_utils import create_hf_model
from evaluations import eval_ScienceQA, eval_MeetingBank, eval_PapyrusF, eval_CStance, eval_Py150, eval_FOMC, eval_NumGLUE_cm, eval_NumGLUE_ds, eval_20Minuten # to be continued
from training.params import Method2Class, AllDatasetName

from model.Replay.LFPT5 import getInitialPrompt
from model.Dynamic_network.PP import PP, convert_PP_model
from model.Dynamic_network.L2P import convert_L2P_model

from moe import NewSdpaAttention, NewLlamaForCausalLM, NewLlamaDecoderLayer, NewLlamaModel

from transformers.models.llama import modeling_llama, LlamaConfig

from lora_callback import global_callback
from peft import peft_model
import types

def copy_module(module):
    new_module = types.ModuleType(module.__name__ + '_original')
    for attr_name in dir(module):
        if not attr_name.startswith('_'):
            attr_value = getattr(module, attr_name)
            setattr(new_module, attr_name, attr_value)
    return new_module

original_modeling_llama = copy_module(modeling_llama)
modeling_llama.LlamaModel = NewLlamaModel
modeling_llama.LlamaForCausalLM = NewLlamaForCausalLM
modeling_llama.LlamaDecoderLayer = NewLlamaDecoderLayer
modeling_llama.LlamaSdpaAttention = NewSdpaAttention


def parse_args():
    def list_of_strings(arg):
        return arg.split(',')
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        type=str,
                        default='',
                        help='Path to the training dataset. A single data path.')
    parser.add_argument('--router_weight_path',
                        type=str,
                        default='',
                        help='Path to the training dataset. A single data path.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--inference_model_path",
        type=str,
        help=
        "Path to inference model.",
        nargs='+',
        required=True,
    )
    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--max_ans_len",
        type=int,
        default=256,
        help="The maximum answer length.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generate temperature params.",
    )

    parser.add_argument(
        "--inference_batch",
        type=int,
        default=1,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--inference_tasks",
        type=list_of_strings,
        default='all',
        help='Datasets to be used.'
    )
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--inference_output_path',
                        type=str,
                        default=None,
                        help="Where to store inference results.")
    parser.add_argument('--CL_method',
            default=None,
            help='continual learning method used')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device("cuda:0")
    inference_tasks = args.inference_tasks 
    task_num = len(inference_tasks)
    inference_model_path = args.inference_model_path
    inference_model_path = inference_model_path[0].split(',')

    def prediction(model, infer_dataloader):
        # global GT_class
        # global predicted_class
        predicted_sequences = []
        sources_sequences = []
        ground_truths = []
        model.eval()
        correct = 0
        count = 0
        for step, batch in enumerate(infer_dataloader):
            global_callback.reset()
            # TODO, add prompts, choosen, rejected
            # implementation, batch = {k: v.to(device) for k, v in batch.items()}
            sources_sequences += batch['sources']
            ground_truths += batch['gts']
            del batch['sources']
            del batch['gts']
            batch = to_device(batch, device)
            prompt_len = batch['input_ids'].shape[1]
            # update progress bar
            progress_bar.update(1)
            description = f"Step {step}"
            progress_bar.set_description(description, refresh=False)
            with torch.no_grad():
                # TODO, add more inference params
                # backbone config
                # generate_ids = model.generate(batch['input_ids'], max_new_tokens=args.max_ans_len,
                #                               pad_token_id=tokenizer.eos_token_id, attention_mask = batch['attention_mask'], temperature=0.7, do_sample=True, repetition_penalty=2.0 )
                # sft config
                generate_ids = model.generate(input_ids=batch['input_ids'],
                                              attention_mask=batch['attention_mask'],
                                              max_new_tokens=args.max_ans_len,
                                              bos_token_id=tokenizer.bos_token_id,
                                              eos_token_id=tokenizer.eos_token_id,
                                              pad_token_id=tokenizer.unk_token_id,
                                              temperature=args.temperature,
                                              do_sample=True,
                                              num_return_sequences=1,
                                              use_cache=True
                                              )
            sequences = tokenizer.batch_decode(generate_ids[:, prompt_len:], skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
            predicted_sequences += sequences

        return sources_sequences, predicted_sequences, ground_truths

    def save_inference_results(evaluation_result: dict, sources_sequences: list, predicted_sequences: list,
                                ground_truths: list, round: int, i_task: int, task: str):
        # save as a json file
        df = {"eval": evaluation_result, 'prompts': sources_sequences, 'results': predicted_sequences,
                'labels': ground_truths}
        if not os.path.exists(args.inference_output_path):
            os.makedirs(args.inference_output_path)
        with open(args.inference_output_path + "/results-" + str(round) + "-" + str(i_task) + "-" + task + ".json", "w+", encoding='utf-8') as file:
            json.dump(df, file, ensure_ascii=False)

    for i in range(0, len(inference_tasks) - 1):
        if i == 0:
            continue
        cur_inference_tasks = inference_tasks[0:i+1]
        all_datasets = []
        for inference_task_id in range(len(cur_inference_tasks)):
            inference_task = inference_tasks[inference_task_id]
            dataset_path = os.path.join(args.data_path, inference_task)
            # Prepare the data
            train, test, infer_dataset = create_prompt_dataset(
                -1,
                dataset_path,
                'dataset_cache',
                42,
                distributed=False
            )

            # infer_dataset = test
            
            infer_dataset.answer_dataset = [inference_task_id for _ in infer_dataset.answer_dataset]
            
            all_datasets.append(infer_dataset)
        
        # continue
        try:
            infer_dataset = torch.utils.data.ConcatDataset(all_datasets)
        except:
            infer_dataset = all_datasets[0]

        if i != 0:
            tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
            model = modeling_llama.LlamaForCausalLM.from_pretrained(args.model_name_or_path, tasks=i+1,torch_dtype=torch.float16)
            fe_weight = torch.load(f'{args.router_weight_path}/step{i-1}_fe_weight.pth', map_location='cpu').to(torch.float16)
            classifier_weight = torch.load(f'{args.router_weight_path}/step{i-1}_router_weight.pth', map_location='cpu').transpose(0, 1).to(torch.float16)
            
            lora_id = 1
            for lora_path in inference_model_path[:i]:
                model.load_adapter(lora_path, adapter_name=f"{lora_id}")
                lora_id += 1
            
            model.model.moe_classifier.weight = torch.nn.Parameter(classifier_weight)
            model.model.fe.weight = torch.nn.Parameter(fe_weight)
        else:
            tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
            model = original_modeling_llama.LlamaForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
            cur_inference_model_path = inference_model_path[0].split(',')[0]
            model.load_adapter(cur_inference_model_path)

        model.to(device)

        inf_data_collator = DataCollator(
            tokenizer,
            model=model,
            padding="longest",
            max_prompt_len=512,
            max_ans_len=256,
            pad_to_multiple_of=8,
            inference=True
        )
        infer_sampler = SequentialSampler(infer_dataset)
        infer_dataloader = DataLoader(infer_dataset,
                                        collate_fn=inf_data_collator,
                                        # sampler=infer_sampler,
                                        shuffle=True,
                                        batch_size=1)
        
        # default the LLM is decoder only model, so padding side is left
        assert tokenizer.padding_side == 'left'
        assert tokenizer.truncation_side == "left"

        inference_model_path = args.inference_model_path
        
        progress_bar = tqdm(total=len(infer_dataloader), leave=True)
        # Inference !
        print("***** Start inference *****", inference_task_id)
        sources_sequences, predicted_sequences, ground_truths = prediction(model, infer_dataloader)
        
        # Get Accuracy/ROUGE/BLEU/...
        # The evaluation result is stored in a dictionary. e.g. {"accuracy": .., "rouge-L": ..}
        if inference_task == "ScienceQA":
            evaluation_result = eval_ScienceQA.eval(predicted_sequences, ground_truths)
        elif inference_task == "MeetingBank":
            evaluation_result = eval_MeetingBank.eval(predicted_sequences, ground_truths)
        elif inference_task == "C-STANCE":
            evaluation_result = eval_CStance.eval(predicted_sequences, ground_truths)
        elif inference_task == "Papyrus-f":
            evaluation_result = eval_PapyrusF.eval(predicted_sequences, ground_truths)
        elif inference_task == "Py150":
            evaluation_result = eval_Py150.eval(predicted_sequences, ground_truths)
        elif inference_task == "FOMC":
            evaluation_result = eval_FOMC.eval(predicted_sequences, ground_truths)
        elif inference_task == "NumGLUE-cm":
            evaluation_result = eval_NumGLUE_cm.eval(predicted_sequences, ground_truths)
        elif inference_task == "NumGLUE-ds":
            evaluation_result = eval_NumGLUE_ds.eval(predicted_sequences, ground_truths)
        elif inference_task == "20Minuten":
            evaluation_result = eval_20Minuten.eval(sources_sequences, predicted_sequences, ground_truths)
        else:
            evaluation_result = {}

        # if args.global_rank <= 0:  # only one process is running
        print("***** Saving inference results *****")
        save_inference_results(evaluation_result, sources_sequences, predicted_sequences, ground_truths, round, inference_task_id, inference_task)
    
if __name__ == "__main__":
    main()
