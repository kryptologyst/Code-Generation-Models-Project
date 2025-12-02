#!/usr/bin/env python3
"""
Evaluation script for code generation models.
"""

import os
import sys
import argparse
import yaml
import torch
import json
import pandas as pd
from typing import List, Dict
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import create_model
from data import CodeDataLoader
from evaluation import CodeEvaluator
from utils import set_seed, get_device, Config

logger = logging.getLogger(__name__)


def evaluate_model(config_path: str, checkpoint_path: str = None):
    """Evaluate a code generation model."""
    # Load configuration
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = Config(config_dict)
    
    # Set seed
    set_seed(config.device.seed)
    
    # Get device
    device = get_device()
    
    # Create model
    model = create_model(config.model.type, config.model.name, device)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.model.load_state_dict(checkpoint['state_dict'])
    
    # Create data loader
    data_loader = CodeDataLoader(
        tokenizer=model.tokenizer,
        batch_size=1,  # Evaluate one at a time for better control
        max_length=config.data.max_length
    )
    
    # Load test dataset
    if not os.path.exists(config.data.test_path):
        logger.error(f"Test dataset not found at {config.data.test_path}")
        return
    
    test_dataset = data_loader.create_dataset(config.data.test_path)
    
    # Create evaluator
    evaluator = CodeEvaluator()
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions = []
    references = []
    prompts = []
    
    for i, item in enumerate(test_dataset):
        if i >= 100:  # Limit evaluation to first 100 samples for speed
            break
        
        prompt = item['prompt']
        reference = item['code']
        
        # Generate code
        generated = model.generate(
            prompt=prompt,
            max_length=config.model.max_length,
            temperature=config.model.temperature,
            top_p=config.model.top_p,
            top_k=config.model.top_k,
            num_return_sequences=1
        )[0]
        
        predictions.append(generated)
        references.append(reference)
        prompts.append(prompt)
        
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1} samples")
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics = evaluator.evaluate_batch(predictions, references, prompts)
    
    # Compute pass@k metrics
    predictions_list = [[pred] for pred in predictions]
    pass_at_k = evaluator.compute_pass_at_k(
        predictions_list, 
        references, 
        config.evaluation.k_values
    )
    metrics.update(pass_at_k)
    
    # Display results
    st.header("Evaluation Results")
    
    # Create results table
    results_data = []
    for metric, value in metrics.items():
        results_data.append({
            'Metric': metric,
            'Value': f"{value:.4f}"
        })
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)
    
    # Save results
    os.makedirs(config.output.results_dir, exist_ok=True)
    results_path = os.path.join(config.output.results_dir, f"{config.model.type}_evaluation_results.json")
    
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    # Save detailed results
    detailed_results = []
    for i, (prompt, pred, ref) in enumerate(zip(prompts, predictions, references)):
        detailed_results.append({
            'index': i,
            'prompt': prompt,
            'prediction': pred,
            'reference': ref
        })
    
    detailed_path = os.path.join(config.output.results_dir, f"{config.model.type}_detailed_results.json")
    with open(detailed_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    logger.info(f"Detailed results saved to {detailed_path}")
    
    return metrics


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Evaluate code generation model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    args = parser.parse_args()
    
    evaluate_model(args.config, args.checkpoint)


if __name__ == '__main__':
    main()
