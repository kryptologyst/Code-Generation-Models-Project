#!/usr/bin/env python3
"""
Quick demonstration of the modernized code generation system.
"""

import os
import sys
import yaml
import torch
from typing import List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models import create_model
from data import create_toy_dataset, save_dataset
from evaluation import CodeEvaluator
from utils import set_seed, get_device, Config

def main():
    """Run a quick demonstration of the code generation system."""
    print("🚀 Code Generation Models - Quick Demo")
    print("=" * 50)
    
    # Set up
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    
    # Create toy dataset
    print("\n📊 Creating toy dataset...")
    data = create_toy_dataset(50)
    os.makedirs('data', exist_ok=True)
    save_dataset(data, 'data/demo.json')
    print(f"Created dataset with {len(data)} samples")
    
    # Test different models
    models_to_test = [
        ("gpt2", "gpt2"),
        ("codet5", "Salesforce/codet5-base"),
        ("codegen", "Salesforce/codegen-350M-mono")
    ]
    
    test_prompts = [
        "Write a Python function that calculates the factorial of a number",
        "Create a function to reverse a string",
        "Implement a binary search algorithm"
    ]
    
    evaluator = CodeEvaluator()
    
    for model_type, model_name in models_to_test:
        print(f"\n🤖 Testing {model_type.upper()} model...")
        
        try:
            # Create model
            model = create_model(model_type, model_name, device)
            print(f"Model loaded successfully")
            
            # Generate samples
            print("Generating code samples...")
            all_predictions = []
            all_references = []
            
            for prompt in test_prompts:
                # Find reference code for this prompt
                reference = None
                for item in data:
                    if item['prompt'] == prompt:
                        reference = item['code']
                        break
                
                if reference:
                    # Generate code
                    generated = model.generate(
                        prompt=prompt,
                        max_length=150,
                        temperature=0.7,
                        top_p=0.9,
                        num_return_sequences=1
                    )[0]
                    
                    all_predictions.append(generated)
                    all_references.append(reference)
                    
                    print(f"\nPrompt: {prompt}")
                    print(f"Generated: {generated[:100]}...")
                    print(f"Reference: {reference[:100]}...")
            
            # Evaluate
            if all_predictions and all_references:
                metrics = evaluator.evaluate_batch(all_predictions, all_references)
                print(f"\n📈 Evaluation Results for {model_type.upper()}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
            
        except Exception as e:
            print(f"❌ Error with {model_type}: {str(e)}")
    
    print("\n✅ Demo completed!")
    print("\nTo run the full system:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Download datasets: python scripts/download_datasets.py")
    print("3. Train a model: python scripts/train.py --config configs/gpt2_codegen.yaml")
    print("4. Run interactive demo: streamlit run demo/app.py")
    print("5. Evaluate model: python scripts/evaluate.py --config configs/gpt2_codegen.yaml")

if __name__ == '__main__':
    main()
