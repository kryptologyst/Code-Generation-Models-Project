#!/usr/bin/env python3
"""
Download datasets for code generation.
"""

import os
import sys
import argparse
import json
import requests
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def download_human_eval():
    """Download HumanEval dataset."""
    url = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse JSONL
        data = []
        for line in response.text.strip().split('\n'):
            if line:
                data.append(json.loads(line))
        
        # Convert to our format
        converted_data = []
        for item in data:
            converted_data.append({
                'prompt': item['prompt'],
                'code': item['canonical_solution'],
                'language': 'python'
            })
        
        # Save to data directory
        os.makedirs('data', exist_ok=True)
        with open('data/human_eval.json', 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Downloaded HumanEval dataset with {len(converted_data)} samples")
        
    except Exception as e:
        logger.error(f"Failed to download HumanEval dataset: {e}")


def download_code_search_net():
    """Download CodeSearchNet dataset (subset)."""
    # This is a simplified version - in practice, you'd want to download the full dataset
    logger.info("CodeSearchNet dataset download not implemented in this demo")
    logger.info("Please download manually from: https://github.com/github/CodeSearchNet")


def create_synthetic_dataset():
    """Create a synthetic dataset for testing."""
    from src.data import create_toy_dataset, save_dataset
    
    # Create larger synthetic dataset
    data = create_toy_dataset(1000)
    
    # Split into train/val/test
    train_data = data[:800]
    val_data = data[800:900]
    test_data = data[900:]
    
    # Save splits
    os.makedirs('data', exist_ok=True)
    save_dataset(train_data, 'data/train.json')
    save_dataset(val_data, 'data/val.json')
    save_dataset(test_data, 'data/test.json')
    
    logger.info("Created synthetic dataset with train/val/test splits")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Download datasets for code generation')
    parser.add_argument('--dataset', type=str, choices=['human_eval', 'code_search_net', 'synthetic'], 
                       default='synthetic', help='Dataset to download')
    args = parser.parse_args()
    
    if args.dataset == 'human_eval':
        download_human_eval()
    elif args.dataset == 'code_search_net':
        download_code_search_net()
    elif args.dataset == 'synthetic':
        create_synthetic_dataset()
    
    logger.info("Dataset download completed!")


if __name__ == '__main__':
    main()
