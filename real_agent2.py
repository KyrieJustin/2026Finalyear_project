import os
import torch
import json
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from real_tools import RealTools

# Configuration Section
QWEN_PATH = "/root/autodl-tmp/huggingface/hub/Qwen2.5-VL-7B-Instruct" 
DINO_PATH = "IDEA-Research/grounding-dino-base"
DEVICE = "cuda"

class RealAgentPipeline:
    def __init__(self):
        print(">>> Loading Models for Real Agent...")
        # Load DINO model
        self.dino_processor = AutoProcessor.from_pretrained(DINO_PATH)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_PATH).to(DEVICE)
        
        # Load Qwen model
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.qwen_processor = AutoProcessor.from_pretrained(QWEN_PATH)
        print(">>> Models Loaded.")

    def detect_and_crop(self, image):
        inputs = self.dino_processor(images=image, text="main object", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
        results = self.dino_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, target_sizes=[image.size[::-1]]
        )[0]
        
        scores = results['scores'].cpu()
        threshold = 0.35
        valid_indices = torch.where(scores > threshold)[0]
        
        if len(valid_indices) > 0:
            best_idx = valid_indices[torch.argmax(scores[valid_indices])]
            box = results['boxes'][best_idx].cpu().numpy()
            x1, y1, x2, y2 = box
            w, h = image.size
            crop = image.crop((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
            return crop
        else:
            return image

    def run_inference(self, image_path, question, use_pinpointing=True, use_knowledge=True):
        image = Image.open(image_path).convert("RGB")
        
         # 1. 处理视觉定位 (Ablation: use_pinpointing)
        if use_pinpointing:
            working_image = self.detect_and_crop(image)
        else:
            working_image = image # 不裁剪，直接用原图


        # Step 1: Visual Enhancement (Crop)
        #cropped_image = self.detect_and_crop(image)
        
        # Step 2: Coarse Identification
        identify_messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": working_image},
                {"type": "text", "text": "Identify the main object in this image specifically. Give me just the name (e.g., 'Boeing 747' or 'Labrador Retriever')."}
            ]
        }]
        text_in = self.qwen_processor.apply_chat_template(identify_messages, tokenize=False, add_generation_prompt=True)
        img_in, _ = process_vision_info(identify_messages)
        inputs = self.qwen_processor(text=[text_in], images=img_in, padding=True, return_tensors="pt").to(self.qwen_model.device)
        with torch.no_grad():
            output_ids = self.qwen_model.generate(**inputs, max_new_tokens=50)
            coarse_label = self.qwen_processor.batch_decode(output_ids, skip_special_tokens=True)[0].split("assistant\n")[-1].strip()
        
        print(f"   [?? AI Guess] I think it is: {coarse_label}")
        
        # Step 3: Real Knowledge Retrieval (Real RAG)
        #knowledge_context = RealTools.get_knowledge(coarse_label, question)
        if use_knowledge:
            knowledge_context = RealTools.get_knowledge(coarse_label, question)
        else:
            knowledge_context = "No external knowledge provided." # 不给外部知识



        # Step 4: Final Inference
        final_prompt = f"""
        User Question: {question}
        
        Preliminary Identification: {coarse_label}
        
        External Knowledge:{knowledge_context}
        
        Instruction:
        Answer the question based on the provided image and information.
        """
        
        final_messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": working_image},
                {"type": "text", "text": final_prompt}
            ]
        }]
        
        text_in = self.qwen_processor.apply_chat_template(final_messages, tokenize=False, add_generation_prompt=True)
        img_in, _ = process_vision_info(final_messages)
        inputs = self.qwen_processor(text=[text_in], images=img_in, padding=True, return_tensors="pt").to(self.qwen_model.device)
        
        with torch.no_grad():
            output_ids = self.qwen_model.generate(**inputs, max_new_tokens=512)
            final_response = self.qwen_processor.batch_decode(output_ids, skip_special_tokens=True)[0].split("assistant\n")[-1].strip()
            
        return final_response

if __name__ == "__main__":
    pass