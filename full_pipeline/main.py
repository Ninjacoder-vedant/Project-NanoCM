#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen 3 4B - Memory-Efficient Full Dataset Training Pipeline
============================================================
Handles ENTIRE CoT dataset (A, B, C problems) with:
- Streaming data loading (no RAM overflow)
- Curriculum learning (A → B → C progression)
- Efficient memory management
- Resume capability
- Mixed difficulty sampling

Author: Enhanced Memory-Efficient Pipeline
"""
import os

def install_dependencies():
    packages = [
        "unsloth",
        "trl==0.24.0",
        "datasets",
        "transformers==5.5.0",
        "bitsandbytes",
        "accelerate",
        "vllm",
        "huggingface_hub",
        "mergekit",
        "flashinfer-cubin==0.6.4",
        "wandb",
        "nltk",
    ]
    for package in packages:
        os.system(f"pip install {package}")

# Uncomment to install dependencies
# install_dependencies()

import os
import re
import sys
import json
import gc
import tempfile
import subprocess
from typing import List, Dict, Tuple, Optional, Iterator
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
import torch

import gc
gc.collect()
torch.cuda.empty_cache()

# ==========================================
# IMPORTS
# ==========================================
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset, Dataset, IterableDataset
from trl import SFTTrainer, GRPOConfig, GRPOTrainer
from transformers import TrainingArguments, TrainerCallback
from transformers.utils import logging
from itertools import islice

logging.set_verbosity_info()

# ==========================================
# MEMORY-EFFICIENT CONFIGURATION
# ==========================================
@dataclass
class MemoryEfficientConfig:
    """Configuration optimized for large dataset training"""
    
    # Model
    model_name: str = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    
    # LoRA configuration (stable settings)
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    
    # Dataset - STREAMING MODE
    dataset_name: str = "open-r1/codeforces-cots"
    dataset_subset: str = "solutions_decontaminated"  # Changed to use C++
    use_streaming: bool = True
    
    # Multi-difficulty training
    difficulty_levels: List[str] = None  # Will be set in __post_init__
    
    # SFT - Curriculum Learning
    sft_samples_per_difficulty: Dict[str, int] = None
    sft_epochs: int = 2
    sft_batch_size: int = 16   # Smashes data through 16 at a time
    sft_grad_accum: int = 2    # Effective batch = 32
    sft_learning_rate: float = 2e-4
    sft_warmup_ratio: float = 0.1
    
    # GRPO - A100 MAX MODE
    grpo_samples_per_difficulty: Dict[str, int] = None
    grpo_epochs: int = 2
    grpo_batch_size: int = 4   # 4 parallel prompts at once
    grpo_grad_accum: int = 2   # Effective batch = 8
    grpo_learning_rate: float = 3e-6
    grpo_num_generations: int = 4  # HUGE: Generates 16 reasoning variations per prompt (takes lots of VRAM but makes RL insanely smart)
    grpo_max_completion_length: int = 4096  # Ample space for long-chain reasoning
    
    # Memory management
    clear_cache_every_n_steps: int = 50
    max_dataset_cache_size: int = 1000  # Max samples to keep in memory
    
    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 2  # Keep only 2 checkpoints
    resume_from_checkpoint: Optional[str] = None
    
    # Paths
    sft_output_dir: str = "outputs/sft_full_dataset"
    grpo_output_dir: str = "outputs/grpo_full_dataset"
    final_model_dir: str = "outputs/final_model_full"
    benchmark_dir: str = "outputs/benchmarks_full"
    
    # System
    seed: int = 3407
    report_to: str = "wandb"  # Visualizing with Weights & Biases
    
    def __post_init__(self):
        """Initialize difficulty-specific settings"""
        if self.difficulty_levels is None:
            self.difficulty_levels = ["A", "B", "C"]
        
        if self.sft_samples_per_difficulty is None:
            # Progressive sampling: more easy, fewer hard
            self.sft_samples_per_difficulty = {
                "A": 8000,   # Most samples from easy problems
                "B": 5000,   # Medium samples from medium problems
                "C": 2000,   # Fewer samples from hard problems
            }
        
        if self.grpo_samples_per_difficulty is None:
            self.grpo_samples_per_difficulty = {
                "A": 2000,
                "B": 1500,
                "C": 500,
            }

config = MemoryEfficientConfig()

# ==========================================
# MEMORY MANAGEMENT UTILITIES
# ==========================================
class MemoryManager:
    """Utilities for managing memory during training"""
    
    @staticmethod
    def clear_cache():
        """Clear GPU and Python cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    @staticmethod
    def print_memory_stats():
        """Print current memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    @staticmethod
    def optimize_model_for_inference(model):
        """Prepare model for inference (saves memory)"""
        model.eval()
        for param in model.parameters():
            param.grad = None
        MemoryManager.clear_cache()

class MemoryEfficientCallback(TrainerCallback):
    """Callback to clear cache periodically"""
    
    def __init__(self, clear_every_n_steps: int = 50):
        self.clear_every_n_steps = clear_every_n_steps
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.clear_every_n_steps == 0:
            MemoryManager.clear_cache()

# ==========================================
# STREAMING DATA LOADER
# ==========================================
class StreamingDataLoader:
    """Memory-efficient data loader using streaming"""
    
    def __init__(self, config: MemoryEfficientConfig):
        self.config = config
    
    def load_streaming_dataset(
        self,
        difficulty: str,
        max_samples: Optional[int] = None
    ) -> Iterator[Dict]:
        """
        Load dataset in streaming mode (never loads full dataset into RAM)
        
        Args:
            difficulty: "A", "B", or "C"
            max_samples: Maximum samples to yield (None = unlimited)
        
        Yields:
            Dataset samples one at a time
        """
        print(f"📡 Streaming difficulty {difficulty} problems...")
        
        # Load in streaming mode
        ds_stream = load_dataset(
            self.config.dataset_name,
            name=self.config.dataset_subset,
            split="train",
            streaming=True
        )
        
        # Filter for difficulty
        filtered = ds_stream.filter(
            lambda x: x['id'].endswith(difficulty)
        )
        
        # Yield samples
        count = 0
        for sample in filtered:
            if max_samples and count >= max_samples:
                break
            yield sample
            count += 1
        
        print(f"✓ Streamed {count} samples for difficulty {difficulty}")
    
    def create_iterable_dataset(
        self,
        difficulty: str,
        max_samples: int,
        format_fn
    ) -> IterableDataset:
        """
        Create an IterableDataset (memory efficient)
        
        This is better than loading all data into a list
        """
        def generator():
            for sample in self.load_streaming_dataset(difficulty, max_samples):
                formatted = format_fn(sample)
                if formatted:
                    yield formatted
        
        return IterableDataset.from_generator(generator)
    
    def load_mixed_difficulty_dataset(
        self,
        samples_per_difficulty: Dict[str, int],
        format_fn,
        shuffle_buffer_size: int = 1000
    ) -> IterableDataset:
        """
        Load dataset with mixed difficulties in memory-efficient way
        
        Args:
            samples_per_difficulty: {"A": 1000, "B": 500, "C": 200}
            format_fn: Function to format each sample
            shuffle_buffer_size: Buffer size for shuffling
        
        Returns:
            IterableDataset with mixed difficulties
        """
        print(f"📚 Loading mixed difficulty dataset:")
        for diff, count in samples_per_difficulty.items():
            print(f"   {diff}: {count} samples")
        
        def generator():
            # Collect samples from each difficulty
            all_samples = []
            
            for difficulty, max_samples in samples_per_difficulty.items():
                for sample in self.load_streaming_dataset(difficulty, max_samples):
                    formatted = format_fn(sample)
                    if formatted:
                        all_samples.append(formatted)
                        
                        # Yield in batches to avoid loading everything
                        if len(all_samples) >= shuffle_buffer_size:
                            np.random.shuffle(all_samples)
                            for s in all_samples:
                                yield s
                            all_samples = []
            
            # Yield remaining samples
            if all_samples:
                np.random.shuffle(all_samples)
                for s in all_samples:
                    yield s
        
        dataset = IterableDataset.from_generator(generator)
        
        # Calculate total samples
        total = sum(samples_per_difficulty.values())
        print(f"✓ Created dataset with {total} total samples")
        
        return dataset

# ==========================================
# CURRICULUM LEARNING SFT
# ==========================================
class CurriculumSFT:
    """SFT with curriculum learning (easy → medium → hard)"""
    
    def __init__(self, config: MemoryEfficientConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.data_loader = StreamingDataLoader(config)
    
    def load_model(self):
        """Load model with memory optimizations"""
        print("="*70)
        print("Loading Model (Memory Optimized)")
        print("="*70)
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
            fast_inference=False,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.config.seed,
            use_rslora=True,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        )
        
        print("✓ Model loaded")
        MemoryManager.print_memory_stats()
    
    def _format_sft_example(self, item: Dict) -> Optional[Dict]:
        """Format example for SFT with CoT structure"""
        try:
            messages = item.get('messages', [])
            if len(messages) < 2:
                return None
            
            system_msg = ""
            user_msg = ""
            assistant_msg = ""
            
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                
                if role == 'system':
                    system_msg = content
                elif role == 'user':
                    user_msg = content
                elif role == 'assistant':
                    assistant_msg = content
            
            if not system_msg:
                system_msg = "You are an expert programmer. Think step-by-step and explain your reasoning."
            
            formatted_text = f"""<|im_start|>system
{system_msg}
<|im_end|>
<|im_start|>user
{user_msg}

Think step-by-step inside <think></think> tags, then provide your solution.
<|im_end|>
<|im_start|>assistant
{assistant_msg}
<|im_end|>"""
            
            return {
                'text': formatted_text,
                'id': item.get('id', 'unknown'),
                'difficulty': item.get('id', '')[-1] if item.get('id') else 'A'
            }
            
        except Exception as e:
            return None
    
    def prepare_curriculum_dataset(self):
        """
        Prepare dataset with curriculum learning
        
        Strategy: Train on mixed difficulties with progressive weighting
        """
        print("\n" + "="*70)
        print("Preparing Curriculum Dataset (Memory Efficient)")
        print("="*70)
        
        # Create mixed difficulty dataset
        train_dataset = self.data_loader.load_mixed_difficulty_dataset(
            samples_per_difficulty=self.config.sft_samples_per_difficulty,
            format_fn=self._format_sft_example,
            shuffle_buffer_size=1000
        )
        
        # Create a comprehensive eval dataset containing A, B, and C
        # This ensures the validation loss tracks all difficulties without slowing down training
        eval_mix = {
            "A": 200,
            "B": 150,
            "C": 50
        }
        eval_dataset = self.data_loader.load_mixed_difficulty_dataset(
            samples_per_difficulty=eval_mix,
            format_fn=self._format_sft_example,
            shuffle_buffer_size=100
        )
        eval_samples = sum(eval_mix.values())
        
        print(f"✓ Training dataset: {sum(self.config.sft_samples_per_difficulty.values())} samples")
        print(f"✓ Eval dataset: {eval_samples} samples")
        
        return train_dataset, eval_dataset
    
    def train_sft(self, train_dataset, eval_dataset):
        """Train SFT with memory optimizations"""
        print("\n" + "="*70)
        print("Starting Curriculum SFT Training")
        print("="*70)
        
        # Calculate max_steps for IterableDataset
        total_samples = sum(self.config.sft_samples_per_difficulty.values())
        effective_batch = self.config.sft_batch_size * self.config.sft_grad_accum
        max_steps = (total_samples // effective_batch) * self.config.sft_epochs
        
        training_args = TrainingArguments(
            max_steps=max_steps,
            # Batch settings (memory optimized)
            per_device_train_batch_size=self.config.sft_batch_size,
            per_device_eval_batch_size=self.config.sft_batch_size,
            gradient_accumulation_steps=self.config.sft_grad_accum,
            num_train_epochs=self.config.sft_epochs,
            
            # Learning
            learning_rate=self.config.sft_learning_rate,
            warmup_ratio=self.config.sft_warmup_ratio,
            lr_scheduler_type="cosine",
            
            # Optimization
            optim="adamw_8bit",
            weight_decay=0.01,
            max_grad_norm=1.0,
            
            # Precision
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            
            # Memory optimizations
            gradient_checkpointing=True,
            
            # Evaluation (less frequent to save time)
            eval_strategy="steps",
            eval_steps=500,
            
            # Logging
            logging_steps=1,  # Log every iteration
            
            # Saving (keep minimal checkpoints)
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            
            # Output
            output_dir=self.config.sft_output_dir,
            report_to=self.config.report_to,
            seed=self.config.seed,
            
            # Resume capability
            resume_from_checkpoint=self.config.resume_from_checkpoint,
        )
        
        # Bypass Unsloth issue with IterableDataset lacking batch_size attribute
        if hasattr(train_dataset, "_ex_iterable") and not hasattr(train_dataset._ex_iterable, "batch_size"):
            train_dataset._ex_iterable.batch_size = self.config.sft_batch_size
        if eval_dataset is not None and hasattr(eval_dataset, "_ex_iterable") and not hasattr(eval_dataset._ex_iterable, "batch_size"):
            eval_dataset._ex_iterable.batch_size = self.config.sft_batch_size
            
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            packing=False,
            args=training_args,
            callbacks=[MemoryEfficientCallback(self.config.clear_cache_every_n_steps)]
        )
        
        print("🚀 Starting training...")
        MemoryManager.print_memory_stats()
        
        trainer_stats = trainer.train(
            resume_from_checkpoint=self.config.resume_from_checkpoint
        )
        
        print("\n💾 Saving model...")
        self.model.save_pretrained(self.config.sft_output_dir + "/final")
        self.tokenizer.save_pretrained(self.config.sft_output_dir + "/final")
        
        # Clear memory
        MemoryManager.clear_cache()
        
        print("✓ SFT Training Complete!")
        return trainer_stats

# ==========================================
# ENHANCED REWARD FUNCTIONS (Same as before)
# ==========================================
class EnhancedRewards:
    """Memory-efficient reward functions"""
    
    @staticmethod
    def format_reward(completions, **kwargs) -> List[float]:
        rewards = []
        for comp in completions:
            content = comp[0]['content'] if isinstance(comp, list) else comp
            score = 0.0
            
            has_think_open = bool(re.search(r"<think>", content))
            has_think_close = bool(re.search(r"</think>", content))
            if has_think_open and has_think_close:
                score += 0.3
            elif has_think_open or has_think_close:
                score += 0.1
            
            has_code = bool(re.search(r"```(python|cpp)\n.*?```", content, re.DOTALL))
            if has_code:
                score += 0.3
            
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            code_match = re.search(r"```(python|cpp)\n(.*?)```", content, re.DOTALL)
            if think_match and code_match:
                if content.index(think_match.group(0)) < content.index(code_match.group(0)):
                    score += 0.2
            
            if len(content.strip()) > 100:
                score += 0.2
            
            rewards.append(score)
        return rewards
    
    @staticmethod
    def reasoning_quality_reward(completions, **kwargs) -> List[float]:
        rewards = []
        for comp in completions:
            content = comp[0]['content'] if isinstance(comp, list) else comp
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if not think_match:
                rewards.append(-0.2)
                continue
            
            thinking = think_match.group(1).strip()
            score = 0.0
            
            word_count = len(thinking.split())
            if word_count < 20:
                score += 0.0
            elif word_count < 50:
                score += 0.1
            elif word_count < 150:
                score += 0.2 + 0.1 * min((word_count - 50) / 100, 1.0)
            elif word_count < 300:
                score += 0.3
            else:
                score += 0.2
            
            logical_patterns = [
                r'\b(first|second|third|then|next|finally)\b',
                r'\b(because|since|therefore|thus|hence)\b',
                r'\b(if|when|while|for each)\b',
                r'\b(consider|note|observe|notice)\b',
            ]
            logic_score = 0.0
            for pattern in logical_patterns:
                matches = len(re.findall(pattern, thinking, re.IGNORECASE))
                logic_score += min(matches * 0.05, 0.075)
            score += min(logic_score, 0.3)
            
            rewards.append(min(score, 1.0))
        return rewards
    
    @staticmethod
    def extract_code(completion: str) -> Tuple[Optional[str], Optional[str]]:
        match = re.search(r"```(python|cpp)\n(.*?)```", completion, re.DOTALL)
        if match:
            return match.group(1), match.group(2).strip()
        return None, None
    
    @staticmethod
    def correctness_reward(completions, input_tests, output_tests, **kwargs) -> List[float]:
        rewards = []
        for comp, inputs, outputs in zip(completions, input_tests, output_tests):
            content = comp[0]['content'] if isinstance(comp, list) else comp
            lang, code = EnhancedRewards.extract_code(content)
            
            if not code or not lang:
                rewards.append(-1.0)
                continue
            
            score = 0
            total = len(inputs)
            compilation_failed = False
            
            for test_input, expected_output in zip(inputs, outputs):
                try:
                    with tempfile.NamedTemporaryFile(
                        mode='w',
                        suffix='.py' if lang == 'python' else '.cpp',
                        delete=False
                    ) as tmp:
                        tmp.write(code)
                        tmp_path = tmp.name
                    
                    try:
                        if lang == 'python':
                            result = subprocess.run(
                                ['python3', tmp_path],
                                input=test_input,
                                capture_output=True,
                                text=True,
                                timeout=5
                            )
                        else:
                            exe_path = tmp_path + '.out'
                            compile_result = subprocess.run(
                                ['g++', '-std=c++17', tmp_path, '-o', exe_path],
                                capture_output=True,
                                text=True,
                                timeout=10
                            )
                            
                            if compile_result.returncode != 0:
                                compilation_failed = True
                                break
                            
                            result = subprocess.run(
                                [exe_path],
                                input=test_input,
                                capture_output=True,
                                text=True,
                                timeout=5
                            )
                            os.remove(exe_path)
                        
                        actual_output = result.stdout.strip()
                        expected_output = str(expected_output).strip()
                        
                        if actual_output == expected_output:
                            score += 1
                    finally:
                        os.remove(tmp_path)
                
                except subprocess.TimeoutExpired:
                    pass
                except Exception:
                    pass
            
            if compilation_failed:
                rewards.append(-2.0)
            elif score == total and total > 0:
                rewards.append(3.0)
            elif score > total / 2:
                partial_reward = 0.5 + 2.0 * (score / total)
                rewards.append(partial_reward)
            else:
                rewards.append(-0.5)
        
        return rewards

# ==========================================
# MEMORY-EFFICIENT GRPO
# ==========================================
class MemoryEfficientGRPO:
    """GRPO with memory optimizations"""
    
    def __init__(self, config: MemoryEfficientConfig, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.rewards = EnhancedRewards()
        self.data_loader = StreamingDataLoader(config)
    
    def _format_grpo_example(self, row: Dict) -> Dict:
        """Format for GRPO"""
        examples = row.get('examples', [])
        input_tests = [str(ex['input']) for ex in examples]
        output_tests = [str(ex['output']) for ex in examples]
        
        prompt = (
            "Think step-by-step in <think></think> tags, then code.\n\n"
            f"{row['prompt']}\n\n<think>\n"
        )
        
        return {
            "prompt": prompt,
            "input_tests": input_tests,
            "output_tests": output_tests,
            "id": row.get('id', 'unknown'),
            "difficulty": row.get('id', '')[-1] if row.get('id') else 'A'
        }
    
    def prepare_grpo_dataset(self):
        """Prepare GRPO dataset with mixed difficulties"""
        print("\n" + "="*70)
        print("Preparing GRPO Dataset (Memory Efficient)")
        print("="*70)
        
        # Load mixed difficulty dataset
        dataset = self.data_loader.load_mixed_difficulty_dataset(
            samples_per_difficulty=self.config.grpo_samples_per_difficulty,
            format_fn=self._format_grpo_example,
            shuffle_buffer_size=500
        )
        
        total = sum(self.config.grpo_samples_per_difficulty.values())
        print(f"✓ GRPO dataset: {total} samples")
        
        return dataset
    
    def train_grpo(self, train_dataset):
        """Train GRPO with memory optimizations"""
        print("\n" + "="*70)
        print("Starting Memory-Efficient GRPO Training")
        print("="*70)
        
        # Calculate max_steps for IterableDataset
        total_samples = sum(self.config.grpo_samples_per_difficulty.values())
        effective_batch = self.config.grpo_batch_size * self.config.grpo_grad_accum
        max_steps = (total_samples // effective_batch) * self.config.grpo_epochs
        
        training_args = GRPOConfig(
            max_steps=max_steps,
            learning_rate=self.config.grpo_learning_rate,
            per_device_train_batch_size=self.config.grpo_batch_size,
            gradient_accumulation_steps=self.config.grpo_grad_accum,
            num_train_epochs=self.config.grpo_epochs,
            
            num_generations=self.config.grpo_num_generations,
            max_completion_length=self.config.grpo_max_completion_length,
            
            loss_type="grpo",
            importance_sampling_level="sequence",
            
            lr_scheduler_type="cosine",
            weight_decay=0.1,
            max_grad_norm=1.0,
            
            logging_steps=1,  # Log every iteration
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            
            output_dir=self.config.grpo_output_dir,
            report_to=self.config.report_to,
        )
        
        # Bypass Unsloth issue with IterableDataset lacking batch_size attribute
        if hasattr(train_dataset, "_ex_iterable") and not hasattr(train_dataset._ex_iterable, "batch_size"):
            train_dataset._ex_iterable.batch_size = self.config.grpo_batch_size
            
        trainer = GRPOTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            reward_funcs=[
                self.rewards.format_reward,
                self.rewards.reasoning_quality_reward,
                self.rewards.correctness_reward,
            ],
            train_dataset=train_dataset,
            callbacks=[MemoryEfficientCallback(self.config.clear_cache_every_n_steps)]
        )
        
        print("🚀 Starting GRPO training...")
        MemoryManager.print_memory_stats()
        
        trainer.train()
        
        print("\n💾 Saving model...")
        trainer.save_model(self.config.grpo_output_dir + "/final")
        self.tokenizer.save_pretrained(self.config.grpo_output_dir + "/final")
        
        MemoryManager.clear_cache()
        print("✓ GRPO Training Complete!")

def generate_with_reasoning_cache(model, tokenizer, prompt_text, max_steps=100, step_tokens=8192):
    """
    Implements a Reasoning Cache scaffold:
    It generates reasoning in chunks. If the model hasn't finished thinking,
    it pauses, summarizes the reasoning so far, and resets the context.
    """
    FastLanguageModel.for_inference(model)
    
    tokenizer.padding_side = "right"
    terminators = [tokenizer.eos_token_id]
    if "<|im_end|>" in tokenizer.get_vocab():
        terminators.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    
    current_context = prompt_text
    full_output = ""
    
    for step in range(max_steps):
        print(f"      [Step {step+1}/{max_steps}] Generating chunk ({step_tokens} tokens max)...")
        
        inputs = tokenizer(current_context, return_tensors="pt").to("cuda")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=step_tokens,
            use_cache=True,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id
        )
        
        input_length = inputs["input_ids"].shape[1]
        chunk_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        full_output += chunk_text
        
        # Stream the model's raw output to console!
        print(f"\n      [RAW MODEL THINKING CHUNK]:\n{'-'*50}\n{chunk_text}\n{'-'*50}\n")
        
        # Check natural stopping criteria: Model finished </think> or outputs code, or hit EOS
        has_finished_thinking = "</think>" in chunk_text
        has_code = "```python" in chunk_text or "```cpp" in chunk_text
        reached_eos = (outputs[0][-1].item() in terminators)
        
        if has_finished_thinking or has_code or reached_eos:
            print("      [✔] Natural stop criteria reached.")
            
            # If it closed the think tag but hasn't output the code yet, generate the code securely
            if has_finished_thinking and not has_code and not reached_eos:
                if "```" not in chunk_text[chunk_text.rfind("</think>"):]:
                    print("      [✔] Thought process finished, generating remaining code...")
                    current_context += chunk_text
                    final_inputs = tokenizer(current_context, return_tensors="pt").to("cuda")
                    final_outputs = model.generate(
                        **final_inputs, max_new_tokens=2000, use_cache=True, temperature=0.3, do_sample=True, repetition_penalty=1.05, eos_token_id=terminators, pad_token_id=tokenizer.eos_token_id
                    )
                    final_text = tokenizer.decode(final_outputs[0][final_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    full_output += final_text
            break
            
        print("      [!] Thinking exceeded chunk limit. Triggering Reasoning Cache summarization...")
        
        # SUMMARIZATION PASS
        summary_prompt = (
            "<|im_start|>system\nYou are an expert assistant managing logical state.<|im_end|>\n"
            f"<|im_start|>user\nHere is a chunk of reasoning for a programming problem:\n\n{chunk_text}\n\n"
            "Summarize this reasoning concisely in 2-3 sentences. Focus strictly on the mathematical state, established facts, and the immediate next logical step. Retain all key details.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        summary_inputs = tokenizer(summary_prompt, return_tensors="pt").to("cuda")
        summary_outputs = model.generate(
            **summary_inputs, max_new_tokens=200, use_cache=True, temperature=0.3, do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
        summary_text = tokenizer.decode(summary_outputs[0][summary_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        print(f"      [Cache Summarized]: {summary_text.strip()[:100]}...")
        
        # FORCED COMPLETION FALLBACK:
        # If we reach the very last step and it STILL hasn't completed, forcefully inject </think>
        if step == max_steps - 1:
            print("      [!] FORCED STOPPING: Reached maximum cache steps. Forcing the model to output the code.")
            current_context = (
                f"{prompt_text}\n"
                "<think>\n"
                f"[Reasoning Cache Final Summary]: {summary_text.strip()}\n"
                "</think>\n"
                "Based on the reasoning summary above, provide the exact final code solution inside fully functional ```python or ```cpp blocks:\n"
            )
            final_inputs = tokenizer(current_context, return_tensors="pt").to("cuda")
            final_outputs = model.generate(
                **final_inputs, max_new_tokens=2000, use_cache=True, temperature=0.3, do_sample=True, eos_token_id=terminators, pad_token_id=tokenizer.eos_token_id
            )
            final_text = tokenizer.decode(final_outputs[0][final_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            full_output += "\n</think>\n" + final_text
            break
        else:
            # ASSEMBLE NEXT CONTEXT 
            # Embed the Cache into the original prompt text
            current_context = (
                f"{prompt_text}\n"
                "<think>\n"
                f"[Reasoning Cache Summary of previous steps]: {summary_text.strip()}\n\n"
                "Continuing reasoning:\n"
            )

    return full_output

# ==========================================
# BENCHMARKING (Simplified for memory)
# ==========================================
class SimpleBenchmark:
    """Lightweight benchmarking"""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.rewards = EnhancedRewards()
        self.data_loader = StreamingDataLoader(config)
    
    def run_quick_benchmark(self, num_samples_per_difficulty: Dict[str, int] = None):
        """Quick benchmark on small sample"""
        if num_samples_per_difficulty is None:
            num_samples_per_difficulty = {"A": 30, "B": 20, "C": 10}
        
        print("\n" + "="*70)
        print("Running Quick Benchmark")
        print("="*70)
        
        results = defaultdict(list)
        
        for difficulty, num_samples in num_samples_per_difficulty.items():
            print(f"\nTesting difficulty {difficulty}...")
            
            samples = list(islice(
                self.data_loader.load_streaming_dataset(difficulty, num_samples),
                num_samples
            ))
            
            for sample in samples:
                formatted = self._format_for_eval(sample)
                if not formatted:
                    continue
                
                completion = self._generate_solution(formatted['prompt'])
                
                # Compute rewards
                format_score = self.rewards.format_reward([completion])[0]
                reasoning_score = self.rewards.reasoning_quality_reward([completion])[0]
                correctness_score = self.rewards.correctness_reward(
                    [completion],
                    [formatted['input_tests']],
                    [formatted['output_tests']]
                )[0]
                
                results[f'{difficulty}_format'].append(format_score)
                results[f'{difficulty}_reasoning'].append(reasoning_score)
                results[f'{difficulty}_correctness'].append(correctness_score)
                
                # Clear memory periodically
                if len(results[f'{difficulty}_format']) % 10 == 0:
                    MemoryManager.clear_cache()
        
        # Print summary
        self._print_results(results)
        
        # Save
        self._save_results(results)
        
        return results
    
    def _format_for_eval(self, sample: Dict) -> Optional[Dict]:
        """Format sample for evaluation"""
        try:
            examples = sample.get('examples', [])
            return {
                'prompt': f"{sample['prompt']}\n\n<think>\n",
                'input_tests': [str(ex['input']) for ex in examples],
                'output_tests': [str(ex['output']) for ex in examples],
            }
        except:
            return None
    
    def _generate_solution(self, prompt: str) -> str:
        """Generate solution using the deep Reasoning Cache."""
        # Strip the trailing <think> tags added by the `_format_for_eval` method
        # because the Reasoning Cache function manages its own <think> embedding natively
        prompt = prompt.replace("\n\n<think>\n", "")
        
        print(f"\n      Starting Deep-Reasoning Inference Pipeline for Benchmark...")
        generated = generate_with_reasoning_cache(
            self.model, 
            self.tokenizer, 
            prompt, 
            max_steps=100, 
            step_tokens=4096
        )
        return generated
    
    def _print_results(self, results: Dict):
        """Print benchmark results"""
        print("\n" + "="*70)
        print("BENCHMARK RESULTS")
        print("="*70)
        
        for difficulty in ["A", "B", "C"]:
            if f'{difficulty}_format' in results:
                fmt = np.mean(results[f'{difficulty}_format'])
                rsn = np.mean(results[f'{difficulty}_reasoning'])
                corr = np.mean(results[f'{difficulty}_correctness'])
                
                print(f"\nDifficulty {difficulty}:")
                print(f"  Format:      {fmt:.3f}")
                print(f"  Reasoning:   {rsn:.3f}")
                print(f"  Correctness: {corr:.3f}")
    
    def _save_results(self, results: Dict):
        """Save results to file"""
        os.makedirs(self.config.benchmark_dir, exist_ok=True)
        output_path = os.path.join(self.config.benchmark_dir, "results.json")
        
        # Convert to serializable format
        serializable = {k: [float(v) for v in vals] for k, vals in results.items()}
        
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"\n✓ Results saved to {output_path}")

# ==========================================
# MAIN PIPELINE
# ==========================================
def main():
    """Execute memory-efficient full dataset pipeline"""
    
    print("\n" + "="*70)
    print(" MEMORY-EFFICIENT FULL DATASET TRAINING PIPELINE")
    print(" Training on A, B, C problems with streaming data")
    print("="*70)
    
    # Configuration
    config = MemoryEfficientConfig()
    
    # print("\n📊 Training Configuration:")
    # print(f"  SFT samples: {sum(config.sft_samples_per_difficulty.values())}")
    # print(f"  GRPO samples: {sum(config.grpo_samples_per_difficulty.values())}")
    # print(f"  Difficulties: {config.difficulty_levels}")
    # print(f"  Memory optimizations: ✓ Streaming, ✓ Cache clearing, ✓ Minimal checkpoints")
    
    # # Step 1: SFT
    # print("\n" + "="*70)
    # print("STEP 1: Curriculum SFT")
    # print("="*70)
    
    # sft_pipeline = CurriculumSFT(config)
    # sft_pipeline.load_model()
    # train_ds, eval_ds = sft_pipeline.prepare_curriculum_dataset()
    # sft_stats = sft_pipeline.train_sft(train_ds, eval_ds)
    
    # Clear memory before GRPO
    MemoryManager.clear_cache()
    
    # Step 2: GRPO
    print("\n" + "="*70)
    print("STEP 2: Memory-Efficient GRPO")
    print("="*70)
    
    grpo_pipeline = MemoryEfficientGRPO(
        config,
        # sft_pipeline.model,
        "./final",
        sft_pipeline.tokenizer
    )
    grpo_ds = grpo_pipeline.prepare_grpo_dataset()
    grpo_pipeline.train_grpo(grpo_ds)
    
    # Clear memory before benchmarking
    MemoryManager.clear_cache()
    
    # Step 3: Benchmark
    print("\n" + "="*70)
    print("STEP 3: Quick Benchmark")
    print("="*70)
    
    benchmark = SimpleBenchmark(
        sft_pipeline.model,
        sft_pipeline.tokenizer,
        config
    )
    results = benchmark.run_quick_benchmark()
    
    # Final save
    print("\n" + "="*70)
    print("Saving Final Model")
    print("="*70)
    
    os.makedirs(config.final_model_dir, exist_ok=True)
    sft_pipeline.model.save_pretrained(config.final_model_dir)
    sft_pipeline.tokenizer.save_pretrained(config.final_model_dir)
    
    print(f"✓ Final model saved to: {config.final_model_dir}")
    
    print("\n" + "="*70)
    print("🎉 FULL DATASET PIPELINE COMPLETE!")
    print("="*70)
    
    MemoryManager.print_memory_stats()
    
    return {
        'sft_stats': sft_stats,
        'benchmark_results': results,
    }

if __name__ == "__main__":
    results = main()
