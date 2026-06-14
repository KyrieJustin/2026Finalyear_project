import json
import os
from tqdm import tqdm
# Initialize the fine-grained pipeline for multi-modal interaction
from simple_agent import FineGrainedPipeline

# --- Configuration Parameters ---
INPUT_JSON = "question.json"               # Input question file path
OUTPUT_JSON = "result_simple_agent.json"   # Output result file path
DATASET_ROOT = "."                         # Dataset root directory (contains bird, car subfolders)

# --- Local Path Search Function ---
# Resolve the actual local path of images referenced in JSON files
# Search from the root directory to find matching image files
def find_local_image_path(json_path_str, root_dir):
    filename = os.path.basename(json_path_str)
    for root, dirs, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

def main():
    # 1. Initialize the multi-modal pipeline
    print(">>> Initializing Pipeline (loading Qwen and DINO models)...")
    try:
        pipeline = FineGrainedPipeline()
    except Exception as e:
        print(f"Model initialization failed, please check model path configuration: {e}")
        return

    # 2. Load JSON question data
    if not os.path.exists(INPUT_JSON):
        print(f"Error: {INPUT_JSON} file not found")
        return
        
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)["qa_pairs"]

    results = []
    print(f"\n>>> Starting to process {len(data)} QA pairs...")

    # 3. Process each QA pair
    for item in tqdm(data):
        original_path = item.get('image_path')
        question = item.get('Question')
        
        # Match the actual local image path
        real_image_path = find_local_image_path(original_path, DATASET_ROOT)
        
        if not real_image_path:
            print(f"\n[Warning] Image not found: {original_path} in current directory")
            item['model_output'] = "Error: Image not found"
            results.append(item)
            continue

        try:
            # Add hint to prompt for multiple-choice questions (output only option + brief explanation)
            prompt_with_hint = question + "\nPlease output the correct option (A, B, C, or D) directly, and briefly explain why."
            
            # Run multi-modal inference
            answer = pipeline.run_inference(real_image_path, prompt_with_hint)
            
            # Save result
            result_item = item.copy()
            result_item['model_output'] = answer
            results.append(result_item)

        except Exception as e:
            print(f"\n[Error] Exception occurred while processing {real_image_path}: {e}")
            item['model_output'] = f"Error: {str(e)}"
            results.append(item)

        # Real-time save results to prevent data loss during processing
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n>>> All tasks completed! Results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()