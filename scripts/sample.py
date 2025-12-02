#!/usr/bin/env python3
"""
Sampling script for code generation models.
"""

import os
import sys
import argparse
import yaml
import torch
import json
from typing import List, Dict
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import create_model
from utils import set_seed, get_device, Config

logger = logging.getLogger(__name__)


def sample_code(
    model,
    prompts: List[str],
    max_length: int = 150,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    num_return_sequences: int = 1
) -> List[List[str]]:
    """Generate code samples for given prompts."""
    all_samples = []
    
    for prompt in prompts:
        samples = model.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences
        )
        all_samples.append(samples)
    
    return all_samples


def save_samples(samples: List[List[str]], prompts: List[str], output_path: str):
    """Save generated samples to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = []
    for prompt, sample_list in zip(prompts, samples):
        for sample in sample_list:
            data.append({
                'prompt': prompt,
                'generated_code': sample
            })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(data)} samples to {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate code samples')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--prompts', type=str, nargs='+', help='Prompts to generate code for')
    parser.add_argument('--prompt_file', type=str, help='File containing prompts (one per line)')
    parser.add_argument('--output', type=str, default='assets/samples/generated_samples.json', help='Output file path')
    parser.add_argument('--max_length', type=int, default=150, help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling parameter')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling parameter')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples per prompt')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = Config(config_dict)
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    
    # Create model
    model = create_model(config.model.type, config.model.name, device)
    
    # Get prompts
    if args.prompts:
        prompts = args.prompts
    elif args.prompt_file:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Default prompts
        prompts = [
            "Write a Python function that calculates the factorial of a number",
            "Create a function to reverse a string",
            "Implement a binary search algorithm",
            "Write a function to check if a number is prime",
            "Create a function to find the maximum element in a list"
        ]
    
    logger.info(f"Generating code for {len(prompts)} prompts...")
    
    # Generate samples
    samples = sample_code(
        model=model,
        prompts=prompts,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_return_sequences=args.num_samples
    )
    
    # Save samples
    save_samples(samples, prompts, args.output)
    
    # Print some examples
    logger.info("Generated samples:")
    for i, (prompt, sample_list) in enumerate(zip(prompts[:3], samples[:3])):
        logger.info(f"\nPrompt {i+1}: {prompt}")
        for j, sample in enumerate(sample_list):
            logger.info(f"Sample {j+1}: {sample[:200]}...")


if __name__ == '__main__':
    main()
