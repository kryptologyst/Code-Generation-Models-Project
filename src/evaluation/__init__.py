"""
Evaluation metrics for code generation models.
"""

import re
import ast
import subprocess
import tempfile
from typing import List, Dict, Optional, Tuple
import numpy as np
from sacrebleu import BLEU
from rouge_score import rouge_scorer
import logging

logger = logging.getLogger(__name__)


class CodeEvaluator:
    """Comprehensive evaluator for code generation models."""
    
    def __init__(self):
        self.bleu_scorer = BLEU()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def evaluate_batch(
        self, 
        predictions: List[str], 
        references: List[str],
        prompts: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate a batch of predictions against references."""
        metrics = {}
        
        # Text-based metrics
        metrics.update(self._compute_bleu(predictions, references))
        metrics.update(self._compute_rouge(predictions, references))
        
        # Code-specific metrics
        metrics.update(self._compute_code_bleu(predictions, references))
        metrics.update(self._compute_execution_success(predictions, references))
        
        # Diversity metrics
        metrics.update(self._compute_diversity_metrics(predictions))
        
        return metrics
    
    def _compute_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute BLEU score."""
        try:
            bleu_score = self.bleu_scorer.corpus_score(predictions, [references])
            return {'bleu': bleu_score.score}
        except Exception as e:
            logger.warning(f"BLEU computation failed: {e}")
            return {'bleu': 0.0}
    
    def _compute_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores."""
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge_scores['rouge1']),
            'rouge2': np.mean(rouge_scores['rouge2']),
            'rougeL': np.mean(rouge_scores['rougeL'])
        }
    
    def _compute_code_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute CodeBLEU score (simplified version)."""
        code_bleu_scores = []
        
        for pred, ref in zip(predictions, references):
            # Extract code blocks
            pred_code = self._extract_code(pred)
            ref_code = self._extract_code(ref)
            
            if pred_code and ref_code:
                # Compute token-level BLEU
                pred_tokens = self._tokenize_code(pred_code)
                ref_tokens = self._tokenize_code(ref_code)
                
                if pred_tokens and ref_tokens:
                    bleu = self._compute_token_bleu(pred_tokens, ref_tokens)
                    code_bleu_scores.append(bleu)
        
        return {'code_bleu': np.mean(code_bleu_scores) if code_bleu_scores else 0.0}
    
    def _compute_execution_success(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute execution success rate."""
        success_count = 0
        total_count = 0
        
        for pred, ref in zip(predictions, references):
            pred_code = self._extract_code(pred)
            ref_code = self._extract_code(ref)
            
            if pred_code and ref_code:
                total_count += 1
                if self._can_execute(pred_code) and self._can_execute(ref_code):
                    success_count += 1
        
        return {'execution_success': success_count / total_count if total_count > 0 else 0.0}
    
    def _compute_diversity_metrics(self, predictions: List[str]) -> Dict[str, float]:
        """Compute diversity metrics."""
        if len(predictions) < 2:
            return {'diversity': 0.0}
        
        # Compute pairwise BLEU scores
        pairwise_bleus = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                bleu = self._compute_token_bleu(
                    self._tokenize_code(self._extract_code(predictions[i])),
                    self._tokenize_code(self._extract_code(predictions[j]))
                )
                pairwise_bleus.append(bleu)
        
        # Diversity is 1 - average pairwise BLEU
        diversity = 1.0 - np.mean(pairwise_bleus) if pairwise_bleus else 0.0
        
        return {'diversity': diversity}
    
    def _extract_code(self, text: str) -> Optional[str]:
        """Extract code from text (look for code blocks or function definitions)."""
        # Look for code blocks in markdown
        code_block_pattern = r'```(?:python)?\n(.*?)```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # Look for function definitions
        function_pattern = r'def\s+\w+\s*\([^)]*\):.*'
        matches = re.findall(function_pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # If no clear code block, return the whole text
        return text.strip()
    
    def _tokenize_code(self, code: str) -> List[str]:
        """Tokenize code into tokens."""
        if not code:
            return []
        
        # Simple tokenization - split on whitespace and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', code)
        return tokens
    
    def _compute_token_bleu(self, pred_tokens: List[str], ref_tokens: List[str]) -> float:
        """Compute BLEU score between token lists."""
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        # Simple BLEU computation
        matches = 0
        for token in pred_tokens:
            if token in ref_tokens:
                matches += 1
        
        precision = matches / len(pred_tokens) if pred_tokens else 0.0
        return precision
    
    def _can_execute(self, code: str) -> bool:
        """Check if code can be executed without errors."""
        if not code:
            return False
        
        try:
            # Try to parse the code
            ast.parse(code)
            
            # Try to execute in a safe environment
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.flush()
                
                # Run the code
                result = subprocess.run(
                    ['python', f.name], 
                    capture_output=True, 
                    timeout=5,
                    text=True
                )
                
                return result.returncode == 0
        except Exception:
            return False
    
    def compute_pass_at_k(
        self, 
        predictions: List[List[str]], 
        references: List[str], 
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """Compute pass@k metrics."""
        pass_at_k = {}
        
        for k in k_values:
            correct_count = 0
            total_count = 0
            
            for pred_list, ref in zip(predictions, references):
                if len(pred_list) >= k:
                    total_count += 1
                    # Check if any of the first k predictions are correct
                    for pred in pred_list[:k]:
                        if self._is_correct(pred, ref):
                            correct_count += 1
                            break
            
            pass_at_k[f'pass@{k}'] = correct_count / total_count if total_count > 0 else 0.0
        
        return pass_at_k
    
    def _is_correct(self, prediction: str, reference: str) -> bool:
        """Check if prediction is functionally correct compared to reference."""
        pred_code = self._extract_code(prediction)
        ref_code = self._extract_code(reference)
        
        if not pred_code or not ref_code:
            return False
        
        # Simple check: if the code can be executed and produces similar output
        try:
            # This is a simplified check - in practice, you'd want more sophisticated testing
            return self._can_execute(pred_code) and self._can_execute(ref_code)
        except Exception:
            return False
