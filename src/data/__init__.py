"""
Data loading and preprocessing utilities for code generation.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


class CodeDataset(Dataset):
    """Dataset class for code generation tasks."""
    
    def __init__(self, data: List[Dict[str, str]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        item = self.data[idx]
        return {
            'prompt': item['prompt'],
            'code': item['code'],
            'language': item.get('language', 'python')
        }


class CodeDataLoader:
    """Data loader for code generation datasets."""
    
    def __init__(self, tokenizer, batch_size: int = 8, max_length: int = 512):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
    
    def create_dataset(self, data_path: str) -> CodeDataset:
        """Create dataset from data file."""
        data = self.load_data(data_path)
        return CodeDataset(data, self.tokenizer, self.max_length)
    
    def load_data(self, data_path: str) -> List[Dict[str, str]]:
        """Load data from various formats."""
        if data_path.endswith('.json'):
            return self.load_json(data_path)
        elif data_path.endswith('.csv'):
            return self.load_csv(data_path)
        elif data_path.endswith('.jsonl'):
            return self.load_jsonl(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
    
    def load_json(self, file_path: str) -> List[Dict[str, str]]:
        """Load data from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'data' in data:
            return data['data']
        else:
            raise ValueError("Invalid JSON format")
    
    def load_csv(self, file_path: str) -> List[Dict[str, str]]:
        """Load data from CSV file."""
        df = pd.read_csv(file_path)
        
        # Expected columns: prompt, code, language (optional)
        required_columns = ['prompt', 'code']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        data = []
        for _, row in df.iterrows():
            item = {
                'prompt': str(row['prompt']),
                'code': str(row['code'])
            }
            if 'language' in df.columns:
                item['language'] = str(row['language'])
            data.append(item)
        
        return data
    
    def load_jsonl(self, file_path: str) -> List[Dict[str, str]]:
        """Load data from JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def create_dataloader(self, dataset: CodeDataset, shuffle: bool = True) -> DataLoader:
        """Create PyTorch DataLoader."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,  # Set to 0 for compatibility
            collate_fn=self.collate_fn
        )
    
    def collate_fn(self, batch: List[Dict[str, str]]) -> Dict[str, Union[List[str], torch.Tensor]]:
        """Collate function for batching."""
        prompts = [item['prompt'] for item in batch]
        codes = [item['code'] for item in batch]
        languages = [item.get('language', 'python') for item in batch]
        
        return {
            'prompts': prompts,
            'codes': codes,
            'languages': languages
        }


def create_toy_dataset(num_samples: int = 100) -> List[Dict[str, str]]:
    """Create a toy dataset for testing purposes."""
    prompts = [
        "Write a Python function that calculates the factorial of a number",
        "Create a function to reverse a string",
        "Implement a binary search algorithm",
        "Write a function to check if a number is prime",
        "Create a function to find the maximum element in a list",
        "Implement a function to sort a list using bubble sort",
        "Write a function to calculate the Fibonacci sequence",
        "Create a function to check if two strings are anagrams",
        "Implement a function to find the greatest common divisor",
        "Write a function to convert Celsius to Fahrenheit"
    ]
    
    codes = [
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
        "def reverse_string(s):\n    return s[::-1]",
        "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
        "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
        "def find_max(lst):\n    return max(lst)",
        "def bubble_sort(lst):\n    n = len(lst)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if lst[j] > lst[j + 1]:\n                lst[j], lst[j + 1] = lst[j + 1], lst[j]\n    return lst",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
        "def are_anagrams(s1, s2):\n    return sorted(s1.lower()) == sorted(s2.lower())",
        "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
        "def celsius_to_fahrenheit(celsius):\n    return (celsius * 9/5) + 32"
    ]
    
    data = []
    for i in range(num_samples):
        prompt_idx = i % len(prompts)
        data.append({
            'prompt': prompts[prompt_idx],
            'code': codes[prompt_idx],
            'language': 'python'
        })
    
    return data


def save_dataset(data: List[Dict[str, str]], file_path: str) -> None:
    """Save dataset to file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if file_path.endswith('.json'):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    elif file_path.endswith('.jsonl'):
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    logger.info(f"Dataset saved to {file_path} with {len(data)} samples")
