import sys
import os
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0].rsplit('/', 2)[0])

import config

os.environ['HF_HOME'] = str(config.DATA_DIR / 'hf_cache')
os.environ['HF_DATASETS_CACHE'] = str(config.DATA_DIR / 'hf_cache')

from datasets import load_dataset
from pathlib import Path
import json


def download_dataset():
    print("Downloading Python code summarization dataset...")
    
    dataset = load_dataset("Nan-Do/code-search-net-python")
    
    raw_data_dir = config.DATA_DIR / "raw"
    raw_data_dir.mkdir(exist_ok=True)
    
    if 'train' in dataset:
        train_data = dataset['train']
        
        total_size = len(train_data)
        train_size = int(total_size * 0.8)
        val_size = int(total_size * 0.1)
        
        train_split = train_data.select(range(train_size))
        val_split = train_data.select(range(train_size, train_size + val_size))
        test_split = train_data.select(range(train_size + val_size, total_size))
        
        splits = {'train': train_split, 'validation': val_split, 'test': test_split}
    else:
        splits = dataset
    
    for split_name, data in splits.items():
        print(f"Processing {split_name} split...")
        
        pairs = []
        for item in data:
            code = item.get('code', item.get('func_code_string', ''))
            summary = item.get('docstring', item.get('func_documentation_string', ''))
            
            if code and summary:
                pairs.append({
                    'code': code,
                    'summary': summary
                })
        
        output_path = raw_data_dir / f"{split_name}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, indent=2)
        
        print(f"Saved {len(pairs)} pairs to {output_path}")
    
    print("Download complete!")


if __name__ == "__main__":
    download_dataset()
