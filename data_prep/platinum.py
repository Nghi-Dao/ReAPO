import os
from datasets import load_dataset

def prepare_platinum_data():
    print("Downloading GSM8K-Platinum...")
    # Load the platinum test set
    ds = load_dataset("madrylab/gsm8k-platinum", split="test")

    # Match the system prompt used in training
    system_prompt = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"

    def process_example(example):
        # Format the prompt just like the MaxRL paper
        question = example["question"] + " Let’s think step by step and output the final answer within \\boxed{}."
        
        # Extract the ground truth answer (everything after ####)
        answer = example["answer"].split("####")[1].strip()

        # Format it into the structure verl expects
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ], 
            "reward_model": {"ground_truth": answer},
            "data_source": "gsm8k-platinum"  # Fixed key
        }

    print("Formatting dataset for verl...")
    # Apply the formatting and remove the old raw columns
    processed_ds = ds.map(process_example, remove_columns=ds.column_names)

    # Save to your data directory
    out_dir = os.path.expanduser("~/data/gsm8k-platinum")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test.parquet")
    
    processed_ds.to_parquet(out_path)
    print(f"✅ Successfully saved GSM8K-Platinum to {out_path}")

if __name__ == "__main__":
    prepare_platinum_data()
