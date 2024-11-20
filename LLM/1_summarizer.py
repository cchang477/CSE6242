import ollama
import pickle
import json
import re
import math
import configparser
import pandas as pd
from tqdm import tqdm
#------------------------------------------------------------------------------------
class Summarizer:
    def __init__(self,config):
        self.filename = config.get('Config','FileName')
        self.dirpath = config.get('Config','RelativeDirectory',fallback='./')
        self.firm_resume = config.getint('Config','FirmIndexResume',fallback=0)
        self.iter_resume = config.getint('Config','IterationResume',fallback=0)
        self.retries = config.getint('Config','RetryAttempts',fallback=5)
        self.write_every = config.getint('Config','WriteEvery',fallback=5)
        self.chunk_size = config.getint('Config','ChunkSize',fallback=5)
        self.model = config.get('Ollama','Model',fallback='llama3.2')
        self.num_ctx = config.getint('Ollama','ContextSize',fallback=2048)
#------------------------------------------------------------------------------------        
    def load_data(self,):
        return pd.read_csv(self.dirpath + self.filename)
        
    def get_range(self,firm_df):
        return range(self.iter_resume,math.ceil(len(firm_df)/5))

    def load_json(self,filepath):
        with open(filepath,'r') as f:
            resp = f.read()
        return resp
    
    def write_json(self,resp,filepath):
        with open(filepath,'w',encoding='utf-8') as f:
            f.write(resp)
    
    def json_to_procons(self,resp):
        resp_dict = json.loads(resp[resp.find('{'):resp.rfind('}')+1]) # load json output string to dictionary
        self.pros = list(resp_dict['pros'].values()) # set pros values to list
        self.cons = list(resp_dict['cons'].values()) # set cons values to list
    
    def prompt_ollama(self,chunk):
        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """
                    You reply in json format like this:
                    {
                        "pros": { "career opportunities": "one positive sentence describing career opportunities else return an empty string",
                                "compensation and benefits": "one positive sentence describing compensation and benefits else return an empty string",
                                "senior management": "one positive sentence describing senior management else return an empty string",
                                "work life balance": "one positive sentence describing work life balance else return an empty string",
                                "culture and values": "one positive sentence describing culture and values else return an empty string",
                                "diversity and inclusion": "one positive sentence describing diversity and inclusion else return an empty string}", 
                        "cons": { "career opportunities": "one negative sentence describing career opportunities else return an empty string",
                                "compensation and benefits": "one negative sentence describing compensation and benefits else return an empty string",
                                "senior management": "one negative sentence describing senior management else return an empty string",
                                "work life balance": "one negative sentence describing work life balance else return an empty string",
                                "culture and values": "one negative sentence describing culture and values else return an empty string",
                                "diversity and inclusion":, "one negative sentence describing diversity and inclusion else return an empty string}"
                    }
                    There should be no escape characters in the output.
                    """,
                },
                {
                    "role": "user",
                    "content": f"""
                    Here are positive reviews {chunk.pros.to_list()+self.pros}.\n
                    Here are negative reviews {chunk.cons.to_list()+self.cons}.""",
                },
            ],
            options={'num_ctx':self.num_ctx} # set context window. default window is 2048 tokens
        )

        return response['message']['content']
#------------------------------------------------------------------------------------    
    def firm_loop(self,firm,df):
        firm_df = df[df.firm == firm]
        rng = self.get_range(firm_df)
        if self.iter_resume!=0:
            resp = self.load_json(f'./output/{self.firm_idx}_{firm}_{self.iter_resume}.txt')
            self.json_to_procons(resp)
        else:
            self.pros = []
            self.cons = []
        self.iter_resume=0
        for i in tqdm(rng):
            for attempt in range(self.retries):
                try:
                    resp = self.iteration_loop(i,firm,firm_df)
                except:
                    print(f'firm: {firm}, iteration: {i}, attempt: {attempt+1}')
                else:
                    break
        self.write_json(resp,f'./output/{self.firm_idx}_{firm}_{i}.txt')
    
    def iteration_loop(self,i,firm,firm_df):
        chunk = firm_df.iloc[i*self.chunk_size:(i*self.chunk_size)+self.chunk_size]
        resp = self.prompt_ollama(chunk)
        if i%(self.write_every)==0:
            self.write_json(resp,f'./output/{self.firm_idx}_{firm}_{i}.txt')
        self.json_to_procons(resp)
        return resp
#------------------------------------------------------------------------------------    
    def main(self):
        df = self.load_data()
        firms = df.firm.unique()
        self.firm_idx=self.firm_resume
        for firm in firms[self.firm_resume:]:
            self.firm_loop(firm,df)
            self.firm_idx+=1
#------------------------------------------------------------------------------------
if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('1_config.ini')
    summarizer = Summarizer(config)
    summarizer.main()