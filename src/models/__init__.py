"""
Code generation models implementation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
import logging

logger = logging.getLogger(__name__)


class CodeGenerationModel(nn.Module):
    """Base class for code generation models."""
    
    def __init__(self, model_name: str, device: torch.device):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
    
    def load_model(self) -> None:
        """Load the pre-trained model and tokenizer."""
        raise NotImplementedError
    
    def generate(
        self, 
        prompt: str, 
        max_length: int = 150, 
        temperature: float = 0.7, 
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True
    ) -> List[str]:
        """Generate code from a prompt."""
        raise NotImplementedError
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for training."""
        raise NotImplementedError


class GPT2CodeGenerator(CodeGenerationModel):
    """GPT-2 based code generator."""
    
    def __init__(self, model_name: str = "gpt2", device: torch.device = torch.device("cpu")):
        super().__init__(model_name, device)
        self.load_model()
    
    def load_model(self) -> None:
        """Load GPT-2 model and tokenizer."""
        logger.info(f"Loading GPT-2 model: {self.model_name}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"GPT-2 model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def generate(
        self, 
        prompt: str, 
        max_length: int = 150, 
        temperature: float = 0.7, 
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True
    ) -> List[str]:
        """Generate code using GPT-2."""
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for training."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


class CodeT5Generator(CodeGenerationModel):
    """CodeT5 based code generator."""
    
    def __init__(self, model_name: str = "Salesforce/codet5-base", device: torch.device = torch.device("cpu")):
        super().__init__(model_name, device)
        self.load_model()
    
    def load_model(self) -> None:
        """Load CodeT5 model and tokenizer."""
        logger.info(f"Loading CodeT5 model: {self.model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"CodeT5 model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def generate(
        self, 
        prompt: str, 
        max_length: int = 150, 
        temperature: float = 0.7, 
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True
    ) -> List[str]:
        """Generate code using CodeT5."""
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for training."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


class CodeGenGenerator(CodeGenerationModel):
    """CodeGen based code generator."""
    
    def __init__(self, model_name: str = "Salesforce/codegen-350M-mono", device: torch.device = torch.device("cpu")):
        super().__init__(model_name, device)
        self.load_model()
    
    def load_model(self) -> None:
        """Load CodeGen model and tokenizer."""
        logger.info(f"Loading CodeGen model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"CodeGen model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def generate(
        self, 
        prompt: str, 
        max_length: int = 150, 
        temperature: float = 0.7, 
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True
    ) -> List[str]:
        """Generate code using CodeGen."""
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for training."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


def create_model(model_type: str, model_name: str, device: torch.device) -> CodeGenerationModel:
    """Factory function to create code generation models."""
    if model_type.lower() == "gpt2":
        return GPT2CodeGenerator(model_name, device)
    elif model_type.lower() == "codet5":
        return CodeT5Generator(model_name, device)
    elif model_type.lower() == "codegen":
        return CodeGenGenerator(model_name, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
