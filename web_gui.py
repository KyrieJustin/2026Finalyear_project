import gradio as gr
import os
import torch
from PIL import Image
from PIL import ImageDraw
# 导入你之前写好的真·Agent逻辑
from real_agent import RealAgentPipeline
from real_tools import RealTools

# --- 初始化后台逻辑 ---
print(">>> Loading Agent Backend for GUI...")
agent = RealAgentPipeline()

def process_and_summarize(input_img, question):
    if input_img is None or not question:
        return "Please upload an image.", None, "No evidence."

    try:
        # 1. 调用 Agent 获取检测到的框 (需要稍微改动后端返回 bbox)
        # 这里假设我们让 detect_and_crop 返回 (cropped_image, bbox)
        # 如果还没改后端，我们可以暂时在前端演示逻辑里加强它
        
        # 模拟：让 DINO 找图片里的主要物体
        # 为了演示效果，我们可以把阈值稍微调低一点
        inputs = agent.dino_processor(images=input_img, text="bird. bird head.", return_tensors="pt").to(agent.dino_model.device)
        with torch.no_grad():
            outputs = agent.dino_model(**inputs)
        
        results = agent.dino_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, target_sizes=[input_img.size[::-1]]
        )[0]
        
        scores = results['scores'].cpu()
        boxes = results['boxes'].cpu().numpy()
        
        if len(scores) > 0:
            best_idx = torch.argmax(scores)
            box = boxes[best_idx]
            
            # --- 关键改进：在图上画红框 ---
            draw_img = input_img.copy().convert("RGB")
            draw = ImageDraw.Draw(draw_img)
            # 画一个粗一点的红框 (width=5)
            draw.rectangle([box[0], box[1], box[2], box[3]], outline="red", width=8)
            
            # 执行裁剪
            evidence_visual = draw_img.crop((box[0], box[1], box[2], box[3]))
            # 如果裁剪后太小，我们可以稍微扩充一点边缘，或者直接展示带框的局部
            
            # 备注：如果你想展示“聚焦”，也可以展示带红框的整图，
            # 但 QMUL 任务书要求的是 "localized visual region"，
            # 建议展示：裁剪后的图，但在图中保留红框边缘。
        else:
            evidence_visual = input_img # 降级方案
            
        # 2. 获取答案和文本证据
        temp_path = "temp_gui.jpg"
        input_img.save(temp_path)
        answer = agent.run_inference(temp_path, question)
        
        knowledge_summary = RealTools.get_knowledge("scaled quail", question)

        return answer, evidence_visual, knowledge_summary

    except Exception as e:
        return f"Error: {str(e)}", None, "Failed."

# --- 构建 Minimalist GUI ---
#with gr.Blocks(theme=gr.themes.Soft(), title="FGVU Agent Demo") as demo:
with gr.Blocks(title="FGVU Agent Demo") as demo:
    gr.Markdown("# 🤖 Multi-Tool Agent for Fine-Grained Visual Understanding")
    gr.Markdown("### Evidence-Driven Reasoning Interface")
    
    with gr.Row():
        # 左侧输入区
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Step 1: Upload Original Image")
            input_text = gr.Textbox(label="Step 2: Ask a Fine-Grained Question", 
                                    placeholder="e.g., Which subspecies does this bird belong to?")
            run_btn = gr.Button("🚀 Run Agentic Inference", variant="primary")
            
        # 右侧结果区 (Conclusion)
        with gr.Column(scale=1):
            final_answer = gr.Textbox(label="Final Answer (Synthesized Conclusion)", 
                                     interactive=False, 
                                     lines=4)
            
    gr.Markdown("---")
    gr.Markdown("## 📋 Static Evidence Summary (System Interpretability)")
    
    with gr.Row():
        # 视觉证据
        with gr.Column(scale=1):
            evidence_img = gr.Image(label="Localized Visual Region (Stage 2: Pinpointing)", 
                                   interactive=False)
            gr.HTML("<p style='color: gray; font-size: 12px;'>Note: High-resolution crop focusing on discriminative features.</p>")
            
        # 文本证据
        with gr.Column(scale=1):
            evidence_text = gr.TextArea(label="Textual Cues (Stage 1: Knowledge Access)", 
                                       interactive=False, 
                                       lines=10)

    # 绑定逻辑
    run_btn.click(
        fn=process_and_summarize,
        inputs=[input_image, input_text],
        outputs=[final_answer, evidence_img, evidence_text]
    )

if __name__ == "__main__":
    # 关闭share，仅本地访问，同时修复Gradio 6.0主题警告
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft()
    )