import torch
from unsloth import FastLanguageModel

def main():
    # 1. Load the SFT Checkpoint
    # Make sure this path points to the exact folder containing your adapter weights
    model_path = "outputs/sft_full_dataset/checkpoint-936"
    print(f"Loading SFT model from: {model_path} ...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    
    # 2. Optimize for inference (makes generation 2x faster and saves VRAM)
    FastLanguageModel.for_inference(model)
    
    # 3. Define a mock Codeforces problem
    user_problem = """You are given an array of N integers. Find the maximum subarray sum.
    
Input:
The first line contains an integer N (1 <= N <= 10^5).
The second line contains N space-separated integers.

Output:
Print a single integer representing the maximum subarray sum."""

    # 4. Format the prompt EXACTLY as it was formatted during SFT phase
    prompt = f"""<|im_start|>system
You are an expert programmer. Think step-by-step and explain your reasoning.
<|im_end|>
<|im_start|>user
{user_problem}

Think step-by-step inside <think></think> tags, then provide your solution.
<|im_end|>
<|im_start|>assistant
"""
    
    # 5. Tokenize and move to GPU
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # 6. Generate the text (Allowing plenty of room for Chain-Of-Thought)
    print("\n" + "="*80)
    print("Generating CoT Response... (This might take a few seconds)")
    print("="*80 + "\n")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=1500, # Large token count so <think> tags aren't cut off
        use_cache=True,
        temperature=0.7,
        top_p=0.9,
    )
    
    # 7. Decode and print
    # We slice out the length of the input prompt so we only print the generation
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    
    final_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(final_output)

if __name__ == "__main__":
    main()
