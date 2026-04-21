import logging
from pathlib import Path
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import Dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig
from transformers.trainer_utils import get_last_checkpoint

# ── Logging ──────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler("logs/train.log"), logging.StreamHandler()],
)
log = logging.getLogger("train")

MODEL_NAME   = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
MAX_SEQ_LEN  = 4096

# ── Model ────────────────────────────────────────────────────────────
log.info(f"Loading {MODEL_NAME}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LEN,
    load_in_4bit    = True,
    full_finetuning = False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r                        = 32,
    target_modules           = ["q_proj","k_proj","v_proj","o_proj",
                                "gate_proj","up_proj","down_proj"],
    lora_alpha               = 32,
    lora_dropout             = 0.05,
    bias                     = "none",
    use_gradient_checkpointing = "unsloth",
    random_state             = 3407,
    use_rslora               = True,
)

SYSTEM_PROMPT = ("You're an expert competitive C++ programmer"
                 "Do all the reasoning before giving code output in "
                 "<think>..</think> tags. Just give the code in ```cpp ``` "
                 "after completing reasoning")

log.info("Loading final_ds.parquet")
full_ds = Dataset.from_parquet("final_ds.parquet")

def format_example(ex):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + list(ex["messages"])
    return {"text": tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False
    )}

full_ds = full_ds.map(format_example, num_proc=4,
                      remove_columns=[c for c in full_ds.column_names if c != "index"])

a_ds = full_ds.filter(lambda x: x["index"] == "A").shuffle(seed=42)
b_ds = full_ds.filter(lambda x: x["index"] == "B").shuffle(seed=42)
c_ds = full_ds.filter(lambda x: x["index"] == "C").shuffle(seed=42)
log.info(f"Sizes — A:{len(a_ds)}  B:{len(b_ds)}  C:{len(c_ds)}")

stage1 = a_ds
stage2 = concatenate_datasets([a_ds, b_ds]).shuffle(seed=42)
stage3 = concatenate_datasets([a_ds, b_ds, c_ds]).shuffle(seed=42)


def make_trainer(ds, output_dir, epochs=1, lr=2e-4):
    trainer = SFTTrainer(
        model            = model,
        processing_class = tokenizer,        # was: tokenizer=
        train_dataset    = ds,
        args = SFTConfig(
            max_seq_length     = MAX_SEQ_LEN,
            dataset_text_field = "text",
            packing = False,   # was False
            per_device_train_batch_size = 32,   # was 8
            gradient_accumulation_steps = 1,    # was 4   (effective batch still 32)
            dataloader_num_workers = 8,      # Utilize more CPU cores for data
            dataloader_pin_memory = True,
            warmup_ratio                = 0.03,
            num_train_epochs            = epochs,
            learning_rate               = lr,
            logging_steps               = 10,
            save_steps                  = 60,
            save_total_limit            = 2,
            optim                       = "adamw_8bit",
            weight_decay                = 0.01,
            lr_scheduler_type           = "cosine",
            seed                        = 3407,
            output_dir                  = output_dir,
            report_to                   = "none",
            bf16                        = is_bfloat16_supported(),
            fp16                        = not is_bfloat16_supported(),
            neftune_noise_alpha         = 5,
        ),
    )
    return train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part    = "<|im_start|>assistant\n",
    )


def run_stage(name, ds, out_dir, epochs, lr):
    last_ckpt = get_last_checkpoint(out_dir) if Path(out_dir).exists() else None
    log.info(f"── {name} ──  examples={len(ds)}  lr={lr}  "
             f"{'resuming from ' + last_ckpt if last_ckpt else 'starting fresh'}")
    trainer = make_trainer(ds, out_dir, epochs=epochs, lr=lr)
    trainer.train(resume_from_checkpoint=last_ckpt)
    log.info(f"── {name} done ──")


run_stage("Stage 1: A",       stage1, "outputs/stage1_A",   epochs=1, lr=2e-4)
run_stage("Stage 2: A+B",     stage2, "outputs/stage2_AB",  epochs=1, lr=1e-4)
run_stage("Stage 3: A+B+C",   stage3, "outputs/stage3_ABC", epochs=1, lr=5e-5)

log.info("Saving final adapter to qwen3_4b_cf_lora")
model.save_pretrained("qwen3_4b_cf_lora")
tokenizer.save_pretrained("qwen3_4b_cf_lora")
log.info("Done.")