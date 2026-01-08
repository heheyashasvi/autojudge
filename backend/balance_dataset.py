
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def balance_dataset(input_path, output_path):
    """
    Balance the dataset by oversampling minority classes.
    """
    logger.info(f"Reading dataset from {input_path}")
    
    # Read the dataset
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                    
    df = pd.DataFrame(data)
    
    # Map difficulty classes if needed (standardize)
    def standardize_class(val):
        if str(val).lower() in ['easy', 'e', '1']: return 'Easy'
        if str(val).lower() in ['medium', 'm', '2']: return 'Medium'
        if str(val).lower() in ['hard', 'h', '3']: return 'Hard'
        return 'Medium' # Default
        
    df['problem_class'] = df['problem_class'].apply(standardize_class)
    
    # Check distribution
    counts = df['problem_class'].value_counts()
    logger.info(f"Original distribution:\n{counts}")
    
    target_count = counts.max()
    logger.info(f"Target count per class: {target_count}")
    
    balanced_dfs = []
    
    for cls in ['Easy', 'Medium', 'Hard']:
        cls_df = df[df['problem_class'] == cls]
        current_count = len(cls_df)
        
        if current_count == 0:
            logger.warning(f"Class {cls} has 0 samples!")
            continue
            
        if current_count < target_count:
            # Oversample
            logger.info(f"Oversampling {cls} from {current_count} to {target_count}")
            # Sample with replacement
            oversampled = cls_df.sample(n=target_count, replace=True, random_state=42)
            balanced_dfs.append(oversampled)
        else:
            # Keep as is (or downsample if we wanted, but we want more data)
            logger.info(f"Keeping {cls} as is ({current_count})")
            balanced_dfs.append(cls_df)
            
    # Combine and shuffle
    balanced_df = pd.concat(balanced_dfs)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Balanced distribution:\n{balanced_df['problem_class'].value_counts()}")
    logger.info(f"Total samples: {len(balanced_df)}")
    
    # Save
    logger.info(f"Saving to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in balanced_df.iterrows():
            record = row.to_dict()
            f.write(json.dumps(record) + '\n')
            
    logger.info("Done!")

if __name__ == "__main__":
    balance_dataset('data/train_dataset.jsonl', 'data/balanced_train.jsonl')
