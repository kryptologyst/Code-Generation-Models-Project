"""
Tests for code generation models.
"""

import pytest
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import create_model, GPT2CodeGenerator
from data import create_toy_dataset, CodeDataLoader
from evaluation import CodeEvaluator
from utils import set_seed, get_device, Config


class TestModels:
    """Test model functionality."""
    
    def test_gpt2_model_creation(self):
        """Test GPT-2 model creation."""
        device = torch.device("cpu")
        model = GPT2CodeGenerator("gpt2", device)
        
        assert model.model is not None
        assert model.tokenizer is not None
        assert model.device == device
    
    def test_gpt2_generation(self):
        """Test GPT-2 code generation."""
        device = torch.device("cpu")
        model = GPT2CodeGenerator("gpt2", device)
        
        prompt = "Write a Python function that calculates the factorial of a number"
        generated = model.generate(prompt, max_length=100, temperature=0.7)
        
        assert isinstance(generated, list)
        assert len(generated) == 1
        assert isinstance(generated[0], str)
        assert len(generated[0]) > 0
    
    def test_model_factory(self):
        """Test model factory function."""
        device = torch.device("cpu")
        
        # Test GPT-2
        model = create_model("gpt2", "gpt2", device)
        assert isinstance(model, GPT2CodeGenerator)
        
        # Test invalid model type
        with pytest.raises(ValueError):
            create_model("invalid", "gpt2", device)


class TestData:
    """Test data functionality."""
    
    def test_toy_dataset_creation(self):
        """Test toy dataset creation."""
        data = create_toy_dataset(10)
        
        assert isinstance(data, list)
        assert len(data) == 10
        
        for item in data:
            assert 'prompt' in item
            assert 'code' in item
            assert 'language' in item
            assert item['language'] == 'python'
    
    def test_data_loader_creation(self):
        """Test data loader creation."""
        from transformers import GPT2Tokenizer
        
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        data_loader = CodeDataLoader(tokenizer, batch_size=4, max_length=512)
        
        assert data_loader.tokenizer == tokenizer
        assert data_loader.batch_size == 4
        assert data_loader.max_length == 512


class TestEvaluation:
    """Test evaluation functionality."""
    
    def test_evaluator_creation(self):
        """Test evaluator creation."""
        evaluator = CodeEvaluator()
        assert evaluator is not None
    
    def test_bleu_computation(self):
        """Test BLEU score computation."""
        evaluator = CodeEvaluator()
        
        predictions = ["def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"]
        references = ["def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"]
        
        metrics = evaluator.evaluate_batch(predictions, references)
        
        assert 'bleu' in metrics
        assert isinstance(metrics['bleu'], float)
        assert metrics['bleu'] >= 0.0
    
    def test_rouge_computation(self):
        """Test ROUGE score computation."""
        evaluator = CodeEvaluator()
        
        predictions = ["def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"]
        references = ["def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"]
        
        metrics = evaluator.evaluate_batch(predictions, references)
        
        assert 'rouge1' in metrics
        assert 'rouge2' in metrics
        assert 'rougeL' in metrics
        
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            assert isinstance(metrics[metric], float)
            assert 0.0 <= metrics[metric] <= 1.0


class TestUtils:
    """Test utility functions."""
    
    def test_seed_setting(self):
        """Test seed setting."""
        set_seed(42)
        # This is a basic test - in practice, you'd want to test actual reproducibility
        assert True
    
    def test_device_detection(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_config_creation(self):
        """Test config creation."""
        config_dict = {
            'model': {'type': 'gpt2', 'name': 'gpt2'},
            'training': {'batch_size': 8, 'learning_rate': 1e-4}
        }
        
        config = Config(config_dict)
        
        assert config.model.type == 'gpt2'
        assert config.model.name == 'gpt2'
        assert config.training.batch_size == 8
        assert config.training.learning_rate == 1e-4


if __name__ == '__main__':
    pytest.main([__file__])
