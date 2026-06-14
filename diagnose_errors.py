import json
import os
import torch
from PIL import Image
from real_agent import RealAgentPipeline
from real_tools import RealTools
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

# --- 配置 ---
RESULT_FILE = "result_real_agent.json"
ERROR_DIR = "error_deep_diagnosis" 
DATASET_ROOT = "."

def run_deep_diagnosis():
    if not os.path.exists(ERROR_DIR): os.makedirs(ERROR_DIR)
    
    # 1. 加载结果和模型
    with open(RESULT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(">>> 正在初始化推理引擎进行深度诊断...")
    agent = RealAgentPipeline()
    
    # 2. 筛选错题
    error_items = [item for item in data if item["Answer"].upper() not in item["model_output"].upper()]
    print(f">>> 发现 {len(error_items)} 个错误案例，开始深度重跑...")

    for i, item in enumerate(tqdm(error_items)):
        if i >= 15: break # 建议先诊断前 15 个，多了 API 容易报错
        
        idx = item.get('index', i)
        filename = os.path.basename(item['image_path'])
        real_path = ""
        for r, d, files in os.walk(DATASET_ROOT):
            if filename in files: real_path = os.path.join(r, filename); break
        
        if not real_path: continue

        try:
            save_folder = os.path.join(ERROR_DIR, f"case_{idx}_{item['Category']}")
            if not os.path.exists(save_folder): os.makedirs(save_folder)
            
            # --- 步骤 1: 重新执行视觉定位 (Crop) ---
            raw_img = Image.open(real_path).convert("RGB")
            cropped_img = agent.detect_and_crop(raw_img)
            raw_img.save(os.path.join(save_folder, "original.jpg"))
            cropped_img.save(os.path.join(save_folder, "agent_crop.jpg"))

            # --- 步骤 2: 重新执行 AI Guess (关键词生成) ---
            identify_messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": cropped_img},
                    {"type": "text", "text": "Identify the main object. Give me just the name."}
                ]
            }]
            text_in = agent.qwen_processor.apply_chat_template(identify_messages, tokenize=False, add_generation_prompt=True)
            img_in, _ = process_vision_info(identify_messages)
            inputs = agent.qwen_processor(text=[text_in], images=img_in, padding=True, return_tensors="pt").to(agent.qwen_model.device)
            with torch.no_grad():
                output_ids = agent.qwen_model.generate(**inputs, max_new_tokens=50)
                coarse_label = agent.qwen_processor.batch_decode(output_ids, skip_special_tokens=True)[0].split("assistant\n")[-1].strip()

            # --- 步骤 3: 重新执行知识检索 ---
            knowledge_context = RealTools.get_knowledge(coarse_label, item['Question'])

            # --- 步骤 4: 执行“话痨版”推理 (强制输出推理过程) ---
            # 修改 Prompt，要求它输出推理过程
            verbose_prompt = f"""
            User Question: {item['Question']}
            Preliminary Identification: {coarse_label}
            External Knowledge: {knowledge_context}
            
            Instruction:
            1. First, analyze if the visual features in the image match the external knowledge.
            2. Show your step-by-step reasoning process.
            3. Finally, provide the correct option (A, B, C, or D).
            """
            
            final_messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": raw_img},
                    {"type": "text", "text": verbose_prompt}
                ]
            }]
            text_in_final = agent.qwen_processor.apply_chat_template(final_messages, tokenize=False, add_generation_prompt=True)
            img_in_final, _ = process_vision_info(final_messages)
            inputs_final = agent.qwen_processor(text=[text_in_final], images=img_in_final, padding=True, return_tensors="pt").to(agent.qwen_model.device)
            with torch.no_grad():
                output_ids_final = agent.qwen_model.generate(**inputs_final, max_new_tokens=1024)
                full_reasoning = agent.qwen_processor.batch_decode(output_ids_final, skip_special_tokens=True)[0].split("assistant\n")[-1].strip()

            # --- 步骤 5: 保存所有证据到日志 ---
            with open(os.path.join(save_folder, "diagnostic_log.txt"), 'w', encoding='utf-8') as f:
                f.write(f"=== CASE #{idx} DIAGNOSIS ===\n")
                f.write(f"Question: {item['Question']}\n")
                f.write(f"Ground Truth: {item['Answer']}\n\n")
                f.write(f"--- Step 1: Visual Logic ---\n")
                f.write(f"Crop Status: Check 'agent_crop.jpg'. If it's the same as original, DINO failed.\n\n")
                f.write(f"--- Step 2: Search Logic ---\n")
                f.write(f"AI Initial Guess: {coarse_label}\n\n")
                f.write(f"--- Step 3: Knowledge Logic ---\n")
                f.write(f"Retrieved Context:\n{knowledge_context}\n\n")
                f.write(f"--- Step 4: Reasoning Logic (Model's Heart) ---\n")
                f.write(f"Model Reasoning Process:\n{full_reasoning}\n")

        except Exception as e:
            print(f"Error diagnosing case {idx}: {e}")

    print(f"\n✅ 深度诊断完成！证据已保存在: {ERROR_DIR}")

if __name__ == "__main__":
    run_deep_diagnosis()