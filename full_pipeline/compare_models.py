import torch
import gc
import re
import os
import subprocess
import tempfile
from datasets import load_dataset
from unsloth import FastLanguageModel

def extract_and_run_code(response: str, input_tests, output_tests):
    """Securely parses markdown code blocks and runs test inputs via native Python/C++ compilers."""
    match = re.search(r"```(python|cpp)\n(.*?)```", response, re.DOTALL)
    if not match:
        return "FAILED: Model failed to provide a properly formatted ```python or ```cpp code block.", 0, len(input_tests)
        
    lang, code = match.group(1), match.group(2).strip()
    score = 0
    total = len(input_tests)
    status_log = f"Language detected: {lang}\n"
    
    if total == 0:
        return status_log + "No Codeforces test cases available to verify.", 0, 0
        
    for i, (t_in, t_out) in enumerate(zip(input_tests, output_tests)):
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py' if lang == 'python' else '.cpp', delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
                
            try:
                if lang == 'python':
                    result = subprocess.run(['python3', tmp_path], input=t_in, capture_output=True, text=True, timeout=5)
                else:
                    exe_path = tmp_path + '.out'
                    compile_result = subprocess.run(['g++', '-std=c++17', tmp_path, '-o', exe_path], capture_output=True, text=True, timeout=10)
                    if compile_result.returncode != 0:
                        return status_log + f"FAILED: C++ Compilation Error\n{compile_result.stderr}", 0, total
                    result = subprocess.run([exe_path], input=t_in, capture_output=True, text=True, timeout=5)
                    os.remove(exe_path)
                    
                actual = result.stdout.strip()
                expected = str(t_out).strip()
                if actual == expected:
                    score += 1
            finally:
                os.remove(tmp_path)
        except subprocess.TimeoutExpired:
            status_log += f"Test {i+1}: Timeout\n"
        except Exception as e:
            status_log += f"Test {i+1}: Runtime Exception {e}\n"
            
    return status_log + f"FINAL SCORE: {score}/{total} TEST CASES PASSED", score, total

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

def main():
    print("="*60)
    print("Loading ONE random 'A' type query from the Codeforces dataset...")
    print("="*60)
    
    # 1. Fetch 1 random A sample natively from HuggingFace
    ds = load_dataset("open-r1/codeforces-cots", "solutions_decontaminated", split="train", streaming=True)
    
    import random
    a_problems = ds.filter(lambda x: str(x.get('id', '')).endswith('A'))
    
    skip_amount = random.randint(0, 100)
    iterator = iter(a_problems)
    for _ in range(skip_amount):
        next(iterator)
        
    prompts = []
    original_problems = []
    test_cases_list = []
    
    row = next(iterator)
    
    # Safely extract problem text
    prompt_text = row.get('prompt', '')
    if not prompt_text and 'messages' in row:
        for msg in row['messages']:
            if msg.get('role') == 'user':
                prompt_text = msg.get('content', '')
                break
                
    original_problems.append(prompt_text)
    
    # Extract inputs/outputs for runtime test cases
    examples = row.get('examples', [])
    input_tests = [str(ex.get('input', '')) for ex in examples]
    output_tests = [str(ex.get('output', '')) for ex in examples]
    test_cases_list.append((input_tests, output_tests))
    
    # Structure the prompt precisely as SFT was trained
    formatted_prompt = f"""<|im_start|>system
You are an expert programmer. Think step-by-step and explain your reasoning.
<|im_end|>
<|im_start|>user
{prompt_text}

Think step-by-step inside <think></think> tags, then provide your solution.
<|im_end|>
<|im_start|>assistant
"""
    prompts.append(formatted_prompt)

    # 2. Base Model Execution
    base_model_name = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
    print("\n" + "="*50)
    print("PHASE 1: INFERENCING INITIAL BASE MODEL")
    print("="*50)
    
    b_model, b_tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=10000,
        load_in_4bit=True,
    )
    base_responses = []
    base_test_results = []
    print("\n   [EVALUATING] Executing compilation & testing for Base Model...")
    for i in range(1):
        print(f"\n   [Problem {i+1}] Generating response with Reasoning Cache...")
        response = generate_with_reasoning_cache(b_model, b_tokenizer, prompts[i], max_steps=100, step_tokens=10000)
        base_responses.append(response)
        
        t_in, t_out = test_cases_list[i]
        log, score, total = extract_and_run_code(response, t_in, t_out)
        base_test_results.append(log)
    
    # 3. Clean Memory safely so A100 doesn't OOM
    del b_model
    del b_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # 4. SFT Model Execution
    sft_model_name = "outputs/sft_full_dataset/checkpoint-936"
    print("\n" + "="*50)
    print("PHASE 2: INFERENCING YOUR SFT MODEL")
    print("="*50)
    
    s_model, s_tokenizer = FastLanguageModel.from_pretrained(
        model_name=sft_model_name,
        max_seq_length=10000,
        load_in_4bit=True,
    )
    sft_responses = []
    sft_test_results = []
    print("\n   [EVALUATING] Executing compilation & testing for SFT Model...")
    for i in range(1):
        print(f"\n   [Problem {i+1}] Generating response with Reasoning Cache...")
        response = generate_with_reasoning_cache(s_model, s_tokenizer, prompts[i], max_steps=100, step_tokens=10000)
        sft_responses.append(response)
        
        t_in, t_out = test_cases_list[i]
        log, score, total = extract_and_run_code(response, t_in, t_out)
        sft_test_results.append(log)
        
    # 5. Output Results to a markdown file natively
    output_params = "comparison_results.md"
    print(f"\nWriting direct comparison logic to '{output_params}' ...")
    
    with open(output_params, "w", encoding="utf-8") as f:
        f.write("# Base Model vs SFT Model Inference Comparison\n\n")
        for i in range(1):
            f.write(f"## Test Case {i+1}\n")
            f.write("<details><summary><b>Click to View Coding Problem</b></summary>\n\n")
            f.write(f"{original_problems[i]}\n\n")
            f.write("</details>\n\n")
            
            f.write("### Base Model (Untrained) Response\n")
            f.write(f"**Test Result:**\n```text\n{base_test_results[i]}\n```\n")
            f.write("<details><summary>View Raw Base Response Tokens</summary>\n\n")
            f.write("```text\n" + base_responses[i] + "\n```\n")
            f.write("</details>\n\n")
            
            f.write("### SFT Trained Response\n")
            f.write(f"**Test Result:**\n```text\n{sft_test_results[i]}\n```\n")
            f.write("<details><summary>View Raw SFT Response Tokens</summary>\n\n")
            f.write("```text\n" + sft_responses[i] + "\n```\n")
            f.write("</details>\n\n")
            f.write("---\n\n")
            
    print("\nDone! Open 'comparison_results.md' to see exactly how your SFT model diverges from the base weights!")

if __name__ == "__main__":
    main()
