import json
import os
import asyncio
from tqdm import tqdm
from real_agent2 import RealAgentPipeline

# --- 配置 ---
INPUT_JSON = "question.json"
DATASET_ROOT = "."

# 定义消融实验的配置
ABLATION_CONFIGS = [
    {
        "name": "vision_only", 
        "use_p": True, 
        "use_k": False, 
        "output": "result_ablation_vision.json"
    },
    {
        "name": "knowledge_only", 
        "use_p": False, 
        "use_k": True, 
        "output": "result_ablation_knowledge.json"
    }
]

def find_local_image_path(json_path_str, root_dir):
    filename = os.path.basename(json_path_str)
    for root, dirs, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

async def run_experiment(pipeline, config, data):
    print(f"\n🚀 Starting Experiment: {config['name']}")
    results = []
    
    for item in tqdm(data, desc=config['name']):
        original_path = item.get('image_path')
        question = item.get('Question')
        real_image_path = find_local_image_path(original_path, DATASET_ROOT)
        
        if not real_image_path:
            continue

        try:
            # 调用带参数的推理函数
            answer = pipeline.run_inference(
                real_image_path, 
                question, 
                use_pinpointing=config['use_p'], 
                use_knowledge=config['use_k']
            )
            
            result_item = item.copy()
            result_item['model_output'] = answer
            results.append(result_item)
            
            # 实时保存，防止中断
            with open(config['output'], 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error in {config['name']}: {e}")

    print(f"✅ Finished: {config['output']}")

async def main():
    # 1. 初始化模型 (只初始化一次，节省显存)
    print(">>> Loading model for Ablation Studies...")
    pipeline = RealAgentPipeline()

    # 2. 读取数据
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)["qa_pairs"]

    # 3. 依次运行消融配置
    for config in ABLATION_CONFIGS:
        # 如果文件已存在，可以选择跳过或者重新跑
        if os.path.exists(config['output']):
            print(f"Skipping {config['name']}, output already exists.")
            continue
            
        await run_experiment(pipeline, config, data)

    print("\n[All Ablation Studies Completed!]")

if __name__ == "__main__":
    asyncio.run(main())



'''
result_baseline.json (纯 Qwen2.5-VL，无工具)
result_ablation_vision.json (Qwen + 视觉定位，无知识) —— 本次生成
result_ablation_knowledge.json (Qwen + 外部知识，无定位) —— 本次生成
result_real_agent.json (完整版 Agent，视觉定位 + 外部知识)
'''