import sys
import os
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0].rsplit('/', 2)[0])

import config

os.environ['HF_HOME'] = str(config.DATA_DIR / 'hf_cache')
os.environ['HF_DATASETS_CACHE'] = str(config.DATA_DIR / 'hf_cache')

import json
import torch
from tqdm import tqdm
from src.data.preprocessing import preprocess_code, preprocess_summary, filter_pair
from src.data.tokenizer import CodeTokenizer


def load_raw_data(split: str) -> list[dict]:
    path = config.DATA_DIR / "raw" / f"{split}.json"
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_dataset():
    print("Loading raw data...")
    train_data = load_raw_data('train')
    val_data = load_raw_data('validation')
    test_data = load_raw_data('test')
    
    print("Preprocessing data...")
    
    def process_split(data, name):
        processed = []
        for item in tqdm(data, desc=f"Processing {name}"):
            code = preprocess_code(item['code'])
            summary = preprocess_summary(item['summary'])
            
            if filter_pair(code, summary):
                processed.append({'code': code, 'summary': summary})
        return processed
    
    train_processed = process_split(train_data, 'train')
    val_processed = process_split(val_data, 'validation')
    test_processed = process_split(test_data, 'test')
    
    print(f"Train: {len(train_processed)}, Val: {len(val_processed)}, Test: {len(test_processed)}")
    
    print("Training tokenizer...")
    all_texts = []
    for item in train_processed:
        all_texts.append(item['code'])
        all_texts.append(item['summary'])
    
    tokenizer = CodeTokenizer()
    tokenizer.train(all_texts)
    print(f"Tokenizer trained with vocab size: {len(tokenizer)}")
    
    def save_split(data, path, name):
        codes = [item['code'] for item in data]
        summaries = [item['summary'] for item in data]
        
        print(f"Tokenizing {name}...")
        code_ids = []
        summary_ids = []
        
        for code, summary in tqdm(zip(codes, summaries), total=len(codes)):
            code_ids.append(tokenizer.encode(code, max_length=config.MAX_CODE_LENGTH))
            summary_ids.append(tokenizer.encode(summary, max_length=config.MAX_SUMMARY_LENGTH))
        
        torch.save({
            'codes': codes,
            'summaries': summaries,
            'code_ids': code_ids,
            'summary_ids': summary_ids
        }, path)
        print(f"Saved {name} to {path}")
    
    save_split(train_processed, config.TRAIN_DATA_PATH, 'train')
    save_split(val_processed, config.VAL_DATA_PATH, 'validation')
    save_split(test_processed, config.TEST_DATA_PATH, 'test')
    
    print("Data preparation complete!")


if __name__ == "__main__":
    prepare_dataset()

