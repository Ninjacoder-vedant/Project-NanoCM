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

    def generate_direct_batched(model, tokenizer, prompts, max_new_tokens=4000):
        """Generates purely direct responses for multiple prompts concurrently."""
        FastLanguageModel.for_inference(model)
        
        # 🌟 A100 BATCH SPEED OPTIMIZATION 🌟
        # Left-padding is required to generate multiple responses concurrently
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        
        print(f"      Running massive parallel matrix calculation for {len(prompts)} prompts...")
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        
        responses = []
        # Because of left-padding, the generated tokens always start precisely at the end of the input_ids shape!
        prompt_length = inputs["input_ids"].shape[1]
        
        for _, output_tensor in enumerate(outputs):
            generated_tokens = output_tensor[prompt_length:]
            decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(decoded)
            
        return responses

    def main():
        print("="*60)
        print("Loading 10 random queries from the Codeforces dataset...")
        print("="*60)
        
        # 1. Fetch 10 random samples natively from HuggingFace
        ds = load_dataset("open-r1/codeforces-cots", "solutions_decontaminated", split="train", streaming=True)
        
        prompts = []
        original_problems = []
        test_cases_list = []
        
        # We will pick the first 10 from the generator stream
        for idx, row in enumerate(ds):
            if idx >= 10: break
            
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
            
            # VERY SIMPLE PROMPT: No "Think step-by-step", no <think> tags.
            formatted_prompt = f"""<|im_start|>system\nYou are an expert programmer.\n<|im_end|>\n<|im_start|>user\n{prompt_text}\n\nProvide your solution in Python or C++.\n<|im_end|>\n<|im_start|>assistant\n"""
            prompts.append(formatted_prompt)

        # 2. Base Model Execution
        base_model_name = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
        print("\n" + "="*50)
        print("PHASE 1: INFERENCING INITIAL BASE MODEL (DIRECT)")
        print("="*50)
        
        b_model, b_tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=10000,
            load_in_4bit=True,
        )
        base_responses = generate_direct_batched(b_model, b_tokenizer, prompts, max_new_tokens=4000)
        base_test_results = []
        print("\n   [EVALUATING] Executing compilation & testing for Base Model...")
        for i in range(len(prompts)):
            print(f"      Testing Case {i+1}...")
            t_in, t_out = test_cases_list[i]
            log, score, total = extract_and_run_code(base_responses[i], t_in, t_out)
            base_test_results.append(log)
        
        # 3. Clean Memory safely so A100 doesn't OOM
        del b_model
        del b_tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        
        # 4. SFT Model Execution
        sft_model_name = "outputs/sft_full_dataset/checkpoint-936"
        print("\n" + "="*50)
        print("PHASE 2: INFERENCING YOUR SFT MODEL (DIRECT)")
        print("="*50)
        
        s_model, s_tokenizer = FastLanguageModel.from_pretrained(
            model_name=sft_model_name,
            max_seq_length=10000,
            load_in_4bit=True,
        )
        sft_responses = generate_direct_batched(s_model, s_tokenizer, prompts, max_new_tokens=4000)
        sft_test_results = []
        print("\n   [EVALUATING] Executing compilation & testing for SFT Model...")
        for i in range(len(prompts)):
            print(f"      Testing Case {i+1}...")
            t_in, t_out = test_cases_list[i]
            log, score, total = extract_and_run_code(sft_responses[i], t_in, t_out)
            sft_test_results.append(log)
            
        # 5. Output Results to a markdown file natively
        output_params = "direct_comparison_results.md"
        print(f"\nWriting direct comparison logic to '{output_params}' ...")
        
        with open(output_params, "w", encoding="utf-8") as f:
            f.write("# Base Model vs SFT Model Direct Inference Comparison (No Thinking)\n\n")
            for i in range(len(prompts)):
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
                
        print(f"\nDone! Open '{output_params}' to see how the models performed WITHOUT thinking tags.")

    if __name__ == "__main__":
        main()
