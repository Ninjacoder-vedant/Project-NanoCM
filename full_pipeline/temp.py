import sys
from main import MemoryEfficientConfig, CurriculumSFT

def main():
    print("Initializing logic from temp.py...")
    # 1. Create the base configuration
    config = MemoryEfficientConfig()
    
    # 2. Initialize the pipeline (we DO NOT load the massive model)
    sft_pipeline = CurriculumSFT(config)
    
    # 3. Request the data generator to fetch some samples
    print("Downloading/Streaming first stream chunk from Hugging Face...")
    train_ds, eval_ds = sft_pipeline.prepare_curriculum_dataset()
    
    print("\n\n" + "🔥" * 35)
    print("   RAW SFT STRING EXACTLY AS THE GPU SEES IT")
    print("🔥" * 35 + "\n")
    
    # 4. Pull exactly 1 sample and print it
    for sample in train_ds:
        print(sample['text'])
        print("\n\n(ID/Difficulty:", sample.get('id'), "->", sample.get('difficulty'), ")")
        break
        
    print("\n" + "="*70)
    print("Notice the <think> tags! This is how CoT is structured for the model.")

if __name__ == "__main__":
    main()
