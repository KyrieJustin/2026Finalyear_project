import json
import os
from tqdm import tqdm
from real_agent import RealAgentPipeline 

INPUT_JSON = "question.json"
OUTPUT_JSON = "result_real_agent.json"
DATASET_ROOT = "."

def find_local_image_path(json_path_str, root_dir):
    filename = os.path.basename(json_path_str)
    for root, dirs, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

def main():
    print(">>> Initializing REAL Agent (With Serper & Wiki)...")
    try:
        pipeline = RealAgentPipeline()
    except Exception as e:
        print(f"Error: {e}")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)["qa_pairs"]

    results = []
    print(f"\n>>> Starting Real-World Evaluation on {len(data)} tasks...")
    print(">>> Note: This will be slower due to API calls.")

    for item in tqdm(data):
        original_path = item.get('image_path')
        question = item.get('Question')
        real_image_path = find_local_image_path(original_path, DATASET_ROOT)
        
        if not real_image_path:
            continue

        try:
            answer = pipeline.run_inference(real_image_path, question)
            
            result_item = item.copy()
            result_item['model_output'] = answer
            results.append(result_item)
            
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"\n[Error] {e}")

if __name__ == "__main__":
    main()