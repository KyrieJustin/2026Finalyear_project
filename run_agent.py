import json
import os
import asyncio
from tqdm.asyncio import tqdm
from fgvc_model import Qwen, FGVCAgent

# --- 配置区域 ---
INPUT_JSON = "question.json"
OUTPUT_JSON = "result_agent.json"
# 注意：根据你的截图，脚本就在 midterm 文件夹里，
# 但图片也在当前目录的子文件夹里 (bird, car...)
# 所以 dataset_root 就是当前目录 "."
DATASET_ROOT = "." 

# 直接使用 HuggingFace ID，利用学校网络下载
QWEN_PATH = "/root/autodl-tmp/huggingface/hub/Qwen2.5-VL-7B-Instruct/"
DINO_PATH = "IDEA-Research/grounding-dino-base"
DEVICE = "cuda"

# --- 智能路径寻找函数 ---
def find_local_image_path(json_path_str, root_dir):
    if not json_path_str: return None
    filename = os.path.basename(json_path_str)
    # 在当前目录递归寻找
    for root, dirs, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

async def main():
    # 1. 初始化模型
    print(">>> 正在初始化 Agent (首次运行会下载模型，请耐心等待)...")
    
    try:
        qwen_instance = Qwen(model_id=QWEN_PATH, device=DEVICE)
        print("✅ Qwen 加载成功。")
    except Exception as e:
        print(f"❌ Qwen 加载失败: {e}")
        return

    try:
        agent = FGVCAgent(ground_model=DINO_PATH, ground_device=DEVICE)
        agent.vlm = qwen_instance 
        print("✅ Agent 初始化成功。")
    except Exception as e:
        print(f"❌ Agent 初始化失败: {e}")
        return

    # 2. 读取数据
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)["qa_pairs"]

    results = []
    print(f"\n>>> 开始运行，共 {len(data)} 条任务...")

    # 3. 异步主循环
    async for item in tqdm(data):
        original_path = item.get('image_path')
        question = item.get('Question')
        
        # 寻找本地真实路径
        real_image_path = find_local_image_path(original_path, DATASET_ROOT)
        
        if not real_image_path:
            print(f"\n[警告] 找不到图片: {original_path}")
            item['model_output'] = "Error: Image not found"
            results.append(item)
            continue

        try:
            # 调用 Agent
            response = await agent(image=real_image_path, text=question)
            
            # 保存结果
            result_item = item.copy()
            result_item['model_output'] = response
            results.append(result_item)
            
            # 实时写入
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"\n[错误] Agent 处理出错: {e}")

    print(f"\n>>> 全部完成！结果已保存至 {OUTPUT_JSON}")

if __name__ == "__main__":
    asyncio.run(main())