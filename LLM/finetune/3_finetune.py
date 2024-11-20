import warnings
import pickle
import argparse
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftConfig, PeftModel
from trl import SFTTrainer
from trl import setup_chat_format
from transformers import (AutoModelForCausalLM, 
                            AutoTokenizer, 
                            BitsAndBytesConfig, 
                            TrainingArguments, 
                            pipeline, 
                            logging,
                            EarlyStoppingCallback, 
                            IntervalStrategy)
from sklearn.metrics import (accuracy_score, 
                            classification_report, 
                            confusion_matrix,
                            recall_score, 
                            precision_score, 
                            f1_score)
from sklearn.model_selection import train_test_split

max_steps = 1500

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('category')
    args = parser.parse_args()

    output_dir=f"../trained_weights/{args.category}"
    
    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    
    with open('../data_dict.pkl','rb') as f:
        data_dict = pickle.load(f)
        
    model_name = "meta-llama/Llama-3.2-3B"

    compute_dtype = getattr(torch, "float16")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=compute_dtype,
        quantization_config=bnb_config, 
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    max_seq_length = 512 #2048
    tokenizer = AutoTokenizer.from_pretrained(model_name, max_seq_length=max_seq_length)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
)

    training_arguments = TrainingArguments(
        output_dir=output_dir,                    # directory to save and repository id
        num_train_epochs=5,                       # number of training epochs
        per_device_train_batch_size=1,            # batch size per device during training
        gradient_accumulation_steps=8,            # number of steps before performing a backward/update pass
        gradient_checkpointing=True,              # use gradient checkpointing to save memory
        optim="paged_adamw_32bit",
        save_steps=0,
        logging_steps=25,                         # log every 10 steps
        learning_rate=2e-4,                       # learning rate, based on QLoRA paper
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
        max_steps=max_steps,
        warmup_ratio=0.03,                        # warmup ratio based on QLoRA paper
        group_by_length=False,
        lr_scheduler_type="cosine",               # use cosine learning rate scheduler
        report_to="tensorboard",                  # report metrics to tensorboard
        #evaluation_strategy="steps",              # save checkpoint every epoch
        #load_best_model_at_end = True,
        #eval_steps = 25,
        #metric_for_best_model = 'accuracy',
    )

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=data_dict['train'][args.category],
        peft_config=peft_config,
        dataset_text_field="prompt",
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        packing=False,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)