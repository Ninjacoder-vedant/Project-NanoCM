import argparse
import logging
import json
import os
import torch
import pandas as pd
from tqdm import tqdm


# ! Accepting Model name from terminal
parser = argparse.ArgumentParser(description="Running this model for eval questions.")
parser.add_argument("--model", type=str, required=True, help="The path of Model on Huggingface")
parser.add_argument("--model_name", type=str, required=True, help="The name of Model on Huggingface")
parser.add_argument("--test_csv", type=str, default="test_problems.parquet", help="Path to test CSV/parquet")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--max_new_tokens", type=int, default=16000)
parser.add_argument("--max_seq_length", type=int, default=20000)
parser.add_argument("--load_in_4bit", action="store_true")
args = parser.parse_args()

model_name = args.model_name
print(f"RESPONSE GENERATION STARTED FOR MODEL : {model_name}")


# ! Loading model with Unsloth
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model,
    max_seq_length=args.max_seq_length,
    dtype=None,              # auto-detect (bf16 on Ampere+, fp16 otherwise)
    load_in_4bit=args.load_in_4bit,
)

# Enable Unsloth's 2x faster inference path
FastLanguageModel.for_inference(model)

# Left padding required for batched generation
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ! Constants
BATCH_SIZE = args.batch_size
MAX_NEW_TOKENS = args.max_new_tokens

base_filepath = os.getcwd()
model_dir_name = model_name.replace("/", "_")

SYSTEM_PROMPT = "Do all the reasoning before giving code output in <think>..</think> tags. Just give the code in ```cpp ``` after completing reasoning"


# ! Logging
log_filename = f"{base_filepath}/pipeline_logs.txt"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def log_message(message):
    print(message, flush=True)
    logging.info(message)

log_message(f"Pipeline started for model: {model_name}")


# ! Load test data
if args.test_csv.endswith(".parquet"):
    df = pd.read_parquet(args.test_csv)
else:
    df = pd.read_csv(args.test_csv)

# Subsample: top N per index
SAMPLE_PER_INDEX = {"A": 10, "B": 5, "C": 5}

df = pd.concat([
    df[df["index"] == k].head(n)
    for k, n in SAMPLE_PER_INDEX.items()
]).reset_index(drop=True)

log_message(f"Sampled {len(df)} problems: " +
            ", ".join(f"{k}={v}" for k, v in df["index"].value_counts().items()))

# Bucket by problem index (A, B, C, ...) — represents difficulty within contest
difficulties = sorted(df["index"].unique().tolist())

import re

CODE_BLOCK_RE = re.compile(r"```cpp\s*\n.*?```", re.DOTALL | re.IGNORECASE)

def has_code_block(response: str) -> bool:
    """Check if response contains a ```cpp ... ``` block."""
    return bool(CODE_BLOCK_RE.search(response))

def build_retry_prompt(original_prompt: str, first_response: str) -> str:
    """Ask the model to finalize code based on its prior reasoning."""
    # Trim the first response to avoid blowing context. Keep last ~4000 chars
    # which usually contains the most recent reasoning / partial code.
    trimmed = first_response[-4000:] if len(first_response) > 4000 else first_response
    return (
        "You previously attempted the following problem but did not produce a final "
        "C++ code block. Here is the original problem and your prior reasoning.\n\n"
        "# Original Problem\n"
        f"{original_prompt}\n\n"
        "# Your Prior Reasoning (may be incomplete)\n"
        f"{trimmed}\n\n"
        "# Task\n"
        "Based on your reasoning above, output ONLY the final C++17 solution inside a single "
        "```cpp\n<code>\n``` block. Do not include <think> tags or additional "
        "explanation — just the code block. "
        "**DON'T START REASONING AGAIN. JUST GIVE THE CODE BASED ON THE PREVIOUS REASONING EVEN THOUGH CODE DOESN'T PASS ALL THE TEST CASES"
    )

def generate_batch_responses(batch_prompts, batch_ids, idx):
    # --- First pass ---
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in batch_prompts
    ]

    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=False,
        add_special_tokens=False,
    ).to("cuda")
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
            pad_token_id=tokenizer.pad_token_id, use_cache=True,
        )
    first_responses = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

    # --- Identify which need retry ---
    retry_indices = [i for i, r in enumerate(first_responses) if not has_code_block(r)]

    second_responses: dict[int, str] = {}
    if retry_indices:
        log_message(f"Retrying {len(retry_indices)}/{len(first_responses)} in batch "
                    f"(index={idx}) — missing code block. IDs: "
                    f"{[batch_ids[i] for i in retry_indices]}")

        retry_prompts = [
            build_retry_prompt(batch_prompts[i], first_responses[i])
            for i in retry_indices
        ]
        retry_texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": rp}],  # no system prompt — just give code
                tokenize=False, add_generation_prompt=True,
            )
            for rp in retry_prompts
        ]

        retry_inputs = tokenizer(
            retry_texts, return_tensors="pt", padding=True, truncation=False,
            add_special_tokens=False,
        ).to("cuda")
        retry_input_len = retry_inputs["input_ids"].shape[1]

        with torch.inference_mode():
            retry_outputs = model.generate(
                **retry_inputs,
                max_new_tokens=4000,  # code-only, shorter budget
                do_sample=True, temperature=0.3, top_p=0.8, top_k=20,  # lower temp for code
                pad_token_id=tokenizer.pad_token_id, use_cache=True,
            )
        decoded_retries = tokenizer.batch_decode(
            retry_outputs[:, retry_input_len:], skip_special_tokens=True
        )
        for i, resp in zip(retry_indices, decoded_retries):
            second_responses[i] = resp
            if has_code_block(resp):
                log_message(f"Retry SUCCESS — id={batch_ids[i]}")
            else:
                log_message(f"Retry STILL MISSING code — id={batch_ids[i]}")

    # --- Save results ---
    responses = []
    for i, (pid, prompt, first_resp) in enumerate(zip(batch_ids, batch_prompts, first_responses)):
        safe_id = pid.replace("/", "_")
        data = {
            "id": pid,
            "index": idx,
            "prompt": prompt,
            "response": first_resp,
            "retry_needed": i in second_responses,
            "retry_response": second_responses.get(i),
            "has_code": has_code_block(first_resp) or (
                i in second_responses and has_code_block(second_responses[i])
            ),
        }
        responses.append(data)

        save_path = f"{base_filepath}/responses/{model_dir_name}/{idx}/{safe_id}/response.json"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        log_message(f"Saved — index={idx}, id={pid}, retry={data['retry_needed']}, "
                    f"has_code={data['has_code']}")

    return responses


# ! Main loop — iterate by difficulty
for difficulty in difficulties:
    file_path = f"{base_filepath}/responses/{model_dir_name}/{difficulty}/response.json"

    if os.path.isfile(file_path):
        log_message(f"Skipping difficulty={difficulty} — already complete for {model_name}")
        continue

    temp_df = df[df["index"] == difficulty].reset_index(drop=True)
    model_responses = []

    # Collect already-processed responses & find first unprocessed index
    work_df = pd.DataFrame()
    for idx, row in temp_df.iterrows():
        pid = row["id"]
        safe_id = pid.replace("/", "_")
        que_file_path = f"{base_filepath}/responses/{model_dir_name}/{difficulty}/{safe_id}/response.json"

        if os.path.isfile(que_file_path):
            with open(que_file_path, "r", encoding="utf-8") as f:
                model_responses.append(json.load(f))
            log_message(f"Already done — difficulty={difficulty}, id={pid}")
        else:
            work_df = temp_df[idx:].reset_index(drop=True)
            break

    # Batch-process remaining
    if not work_df.empty:
        for i in tqdm(range(0, len(work_df), BATCH_SIZE),
                      desc=f"difficulty={difficulty}"):
            batch_df = work_df.iloc[i : i + BATCH_SIZE]
            batch_prompts = batch_df["prompt"].tolist()
            batch_ids = batch_df["id"].tolist()
            try:
                batch_data = generate_batch_responses(batch_prompts, batch_ids, difficulty)
                model_responses.extend(batch_data)
            except torch.cuda.OutOfMemoryError:
                log_message(f"OOM at batch {i} for difficulty={difficulty}. Try smaller --batch_size.")
                torch.cuda.empty_cache()
                raise

    # Save aggregate per difficulty
    response_df = pd.DataFrame(model_responses)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    response_df.to_json(file_path, orient="records", indent=4)

    log_message(f"Difficulty={difficulty} complete → {file_path}")

log_message("Pipeline finished.")