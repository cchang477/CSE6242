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

cat_map = {'co':'career opportunities',
          'cb':'compensation and benefits',
          'sm':'senior management',
          'cv':'culture and values',
          'di':'diversity and inclusion',
          'wlb':'work life balance'}

lbl_map = {'terrible':1, 'bad':2, 'neutral':3, 'none':3, 'good':4, 'excellent':5}

def generate_test_prompt(data_point):
    return f"""
On the topic of {data_point["category_full"]}, analyze the sentiment of the company review enclosed in square brackets,
determine if it is excellent, good, neutral, bad, or terrible, and return the answer as 
the corresponding sentiment label "excellent" or "good" or "neutral" or "bad" or "terrible".

[{data_point["text"]}] = """.strip()

def predict(test, model, tokenizer):
    y_pred = []
    replies = []
    for i in tqdm(range(len(test))):
        prompt = test.iloc[i]["prompt"]
        pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens = 3,
                        #temperature = 0.7,
                        do_sample = False, # equivalent to temperature 0 i.e. deterministic process
                       )
        result = pipe(prompt)
        answer = result[0]['generated_text'].split("=")[-1]
        replies.append(result[0]['generated_text'])
        if "terrible" in answer:
            y_pred.append("terrible")
        elif "bad" in answer:
            y_pred.append("bad")
        elif "neutral" in answer:
            y_pred.append("neutral")
        elif "good" in answer:
            y_pred.append("good")
        elif "excellent" in answer:
            y_pred.append("excellent")
        else:
            y_pred.append("none")
    return y_pred, replies

def evaluate(y_true, y_pred, verbose=False, print_reports=True):
    labels = ['terrible', 'bad', 'neutral', 'good', 'excellent']
    mapping = {'terrible':0, 'bad':1, 'neutral':2, 'none':2, 'good':3, 'excellent':4}
    def map_func(x):
        return mapping.get(x, 1)
    
    y_true = np.vectorize(map_func)(y_true)
    y_pred = np.vectorize(map_func)(y_pred)
    if verbose==True:
        print(f'y_true: {y_true}')
        print(f'y_pred: {y_pred}')
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    
    # Generate accuracy report
    unique_labels = set(y_true)  # Get unique labels
    
    for label in unique_labels:
        label_indices = [i for i in range(len(y_true)) 
                         if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        
    # Generate classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2, 3, 4])
    
    if print_reports==True:
        print(f'Accuracy: {accuracy:.3f}')        
        print(f'Accuracy for label {label}: {accuracy:.3f}')        
        print('\nClassification Report:')
        print(class_report)
        print('\nConfusion Matrix:')
        print(conf_matrix)
    
    return class_report, conf_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('category')
    args = parser.parse_args()

    cat=args.category
    
    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    
    summ = pd.read_csv('../summ_text.csv')
    summ['category_full'] = cat_map[cat]
    summ['prompt'] = summ.apply(generate_test_prompt,axis=1)
    
    # model_name = "meta-llama/Llama-3.2-3B"

    compute_dtype = getattr(torch, "float16")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    
    model_path = f'../trained_weights/{cat}'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=compute_dtype,
        quantization_config=bnb_config, 
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    max_seq_length = 512 #2048
    tokenizer = AutoTokenizer.from_pretrained(model_path, max_seq_length=max_seq_length)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    to_predict = summ.iloc[:]
    y_pred, _ = predict(to_predict, model, tokenizer)

    pd.Series([lbl_map[pred] for pred in y_pred]).to_csv(f'4_summ_ratings_{cat}.csv')
    
    # y_true = data_dict['y_true'][cat][:]
    # evaluation = pd.DataFrame({'prompt': data_dict['test'][cat]['prompt'], 
    #                            'y_true':y_true, 
    #                            'y_pred': y_pred,
    #                            'base_pred':base_eval[cat]['y_pred']},
    #                          )
    # evaluation.to_csv(f"./{cat}_test_predictions.csv", index=False)