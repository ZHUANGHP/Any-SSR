import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

from transformers.models.llama import LlamaForCausalLM
import torch
from torch.nn import CrossEntropyLoss
from typing import Optional, List, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

import logging

from transformers.cache_utils import Cache, DynamicCache

from transformers.utils import logging

from transformers.models.llama import LlamaModel

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaSdpaAttention, LlamaConfig, LlamaRMSNorm, LlamaRotaryEmbedding, LlamaAttention, LlamaFlashAttention2, LlamaMLP, repeat_kv, apply_rotary_pos_emb

import torch.nn as nn

import torch.nn.functional as F

import logging
import transformers

from peft import LoraConfig
import torch

import json
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from peft import get_peft_model

from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import torch.nn.functional as F

LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}
feature_layers = 4
gamma = 10000
router_weights_path = '/home/yanyue/TRACE_anyssr/router_weights'
dataset_path = '/home/yanyue/LLM-CL-Benchmark_5000/'
dataset_cache_path = '/home/yanyue/TRACE/outputs_router'

class NewLlamaForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config, task_number, gamma):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.model.layers = torch.nn.ModuleList(self.model.layers[:4])  # 仅取前4层进行分类

        # self.cls_head = torch.nn.Linear(in_features=4096, out_features=task_number)
        self.fe = torch.nn.Linear(in_features=4096, out_features=gamma)

        self.cls_head = torch.nn.Linear(in_features=gamma, out_features=task_number)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        # 这里由于仅取前四层，outputs应该是第四层的输出

        hidden_mean = outputs[0].mean(dim=1)

        out = self.fe(hidden_mean)

        out = self.cls_head(out)
        return out
    
    def MoeClassifier():
        pass

def load_model_and_tokenizer(step):
    model = NewLlamaForCausalLM.from_pretrained(
                'meta-llama/Llama-2-7b-chat-hf',
                device_map="auto",
                torch_dtype="auto",
                task_number=step+2,
                trust_remote_code=True,
                gamma=gamma
            )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'meta-llama/Llama-2-7b-chat-hf', trust_remote_code=True
    )

    return model, tokenizer

def load_tokenizer():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'meta-llama/Llama-2-7b-chat-hf', trust_remote_code=True
    )

    return tokenizer

def train():
    inference_tasks = ['NumGLUE-cm','NumGLUE-ds','FOMC','20Minuten','C-STANCE','Py150','MeetingBank','ScienceQA']
    import numpy as np

    def eval_router(model, infer_dataloader, step):
        # predicted_sequences = []
        # sources_sequences = []
        # ground_truths = []
        # model.eval()
        fe_weight = torch.load(f'{router_weights_path}/step{step}_fe_weight.pth', map_location=model.device).to(torch.float16)
        classifier_weight = torch.load(f'{router_weights_path}/step{step}_router_weight.pth', map_location=model.device).transpose(0, 1).to(torch.float16)
        model.cls_head.weight = torch.nn.Parameter(classifier_weight)
        model.fe.weight = torch.nn.Parameter(fe_weight)
        with torch.no_grad():
            count = 0
            correct = 0
            print('-----------------------start new epoch-------------------')
            for steps, batch in enumerate(infer_dataloader):
                # TODO, add prompts, choosen, rejected
                # implementation, batch = {k: v.to(device) for k, v in batch.items()}
                # sources_sequences += batch['sources']
                # ground_truths += batch['gts']
                # del batch['sources']
                # del batch['gts']
                # batch = to_device(batch, device)
                # prompt_len = batch['input_ids'].shape[1]
                # update progress bar
                labels = batch['gts']
                input_ids = batch['input_ids']
                input_ids = input_ids.to('cuda')
                prediction = model(input_ids).to(torch.float32)

                if labels == [prediction.argmax().item()]:
                    correct += 1
                else:
                    print(f'prediction: {prediction.argmax().item()}, labels: {labels}')
                
                count += 1
                
            acc = correct / count
            print(f'step{step} has an acc of：{acc}')

    for i in range(0, len(inference_tasks) - 1):
        if i != 6:
            continue
        model, tokenizer = load_model_and_tokenizer(i)
        tokenizer.pad_token = tokenizer.eos_token

        # if i != (len(inference_tasks) - 2):
        #     continue
        cur_inference_tasks = inference_tasks[0:i+2]
        all_datasets = []
        for inference_task_id in range(len(cur_inference_tasks)):    # evaluation for previous tasks in a single round
            inference_task = inference_tasks[inference_task_id]
            cur_dataset_path = os.path.join(dataset_path, inference_task)
            # Prepare the data
            train, test, infer_dataset = create_prompt_dataset(
                -1,
                cur_dataset_path,
                '/home/yanyue/TRACE/outputs_router',
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
        # progress_bar = tqdm(total=len(infer_dataloader), leave=True)

        # Inference !
        print("***** Start inference *****")
        eval_router(model, infer_dataloader, i)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    print(
        "-----------------------------------start training---------------------------------------"
    )
    train()