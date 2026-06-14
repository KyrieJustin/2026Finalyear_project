import json
import os
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# --- 配置区域 ---
INPUT_JSON = "question.json"           
OUTPUT_JSON = "result_baseline.json"   
DATASET_ROOT = "."             
QWEN_PATH = "/root/autodl-tmp/huggingface/hub/Qwen2.5-VL-7B-Instruct"

# 智能寻找本地图片路径
def find_local_image_path(json_path_str, root_dir):
    if not json_path_str: return None
    filename = os.path.basename(json_path_str)
    for root, dirs, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

def main():
    print(f">>> 正在加载 Baseline 模型 (Qwen2.5-VL-7B Vanilla) ...")
    try:
        processor = AutoProcessor.from_pretrained(QWEN_PATH)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_PATH,
            torch_dtype=torch.bfloat16,
            #attn_implementation="flash_attention_2", # 如果报错，可以删掉这行
            device_map="auto"
        )
        print(">>> Baseline 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 读取问题数据
    if not os.path.exists(INPUT_JSON):
        print(f"❌ 错误: 找不到 {INPUT_JSON}")
        return
        
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)["qa_pairs"]

    results =[]
    print(f"\n>>> 开始运行 Baseline (纯图文问答)，共 {len(data)} 条任务...")

    for item in tqdm(data):
        original_path = item.get('image_path')
        question = item.get('Question')
        
        real_image_path = find_local_image_path(original_path, DATASET_ROOT)
        
        if not real_image_path:
            print(f"\n[警告] 找不到图片: {original_path}，跳过。")
            item['model_output'] = "Error: Image not found"
            results.append(item)
            continue

        try:
            # 为了公平对比，使用和 Agent 完全一样的 Prompt 后缀，但不给任何外部知识
            prompt_with_hint = question + "\nPlease output the correct option (A, B, C, or D) directly, and briefly explain why."
            image = Image.open(real_image_path).convert("RGB")
            
            # 只输入一张原图和问题
            messages =[
                {
                    "role": "user",
                    "content":[
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_with_hint}
                    ]
                }
            ]
            
            # 模型推理
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=512)
                output_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]
            
            # 提取回答文本
            response = output_text.split("assistant\n")[-1].strip()
            
            result_item = item.copy()
            result_item['model_output'] = response
            results.append(result_item)

        except Exception as e:
            print(f"\n[错误] 处理 {real_image_path} 时出错: {e}")
            item['model_output'] = f"Error: {str(e)}"
            results.append(item)

        # 实时保存结果
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n>>> Baseline 运行结束！结果已保存至 {OUTPUT_JSON}")

if __name__ == "__main__":
    main()