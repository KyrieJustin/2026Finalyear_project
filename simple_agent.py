import os
import torch
import json
import base64
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Qwen2_5_VLForConditionalGeneration
import numpy as np

# --- 1. Configuration (Modify according to your environment) ---
# Qwen model path
QWEN_PATH = "/root/autodl-tmp/huggingface/hub/Qwen2.5-VL-7B-Instruct" 
# DINO model path (customizable)
DINO_PATH = "IDEA-Research/grounding-dino-base"
DEVICE = "cuda"

# --- 2. Tool Class (Can be integrated into Agent/Runner) ---
class SimpleTools:
    """
    For demonstration, a lightweight knowledge base is built into the model.
    In actual use, you can replace the search_similar_images logic or directly use the built-in logic.
    """
    @staticmethod
    def search_knowledge(label):
        # Actual logic: Connect to Web Search -> Wikipedia filtering logic
        # Demo logic: If the label is found in the knowledge base, return the corresponding description; otherwise return default.
        # During demonstration, you can hardcode a brief description of several categories for demonstration effect.
        
        print(f"   [Tool] Searching knowledge for: {label}...")
        
        # In actual scenarios, replace it with the get_wikipedia_description function connected to the Wikipedia API
        knowledge_base = {
            "bird": "Birds have feathers, a beak, and lay eggs. Key features for identification include beak shape and wing patterns.",
            "dog": "Dogs are domesticated mammals. Breeds are distinguished by ear shape, snout length, and coat texture.",
            "car": "Cars are identified by their grille shape, headlight design, and logo.",
            "default": f"This is a {label}. Detailed analysis requires looking at specific visual features."
        }
        
        # Built-in keyword matching
        for key in knowledge_base:
            if key in label.lower():
                return knowledge_base[key]
        return knowledge_base["default"]
        
        
        
        
        
# Add this function before FineGrainedPipeline class
def process_vision_info(messages):
    """Extract image/video inputs from messages (compatible with Qwen2.5-VL input format)"""
    image_inputs = []
    video_inputs = []
    for message in messages:
        if isinstance(message, dict) and "content" in message:
            for content in message["content"]:
                if content["type"] == "image":
                    image_inputs.append(content["image"])
                elif content["type"] == "video":
                    video_inputs.append(content["video"])
    return image_inputs, video_inputs
    
    
    
    
# --- 3. Model Pipeline (Integrates DINO and Qwen) ---
class FineGrainedPipeline:
    def __init__(self):
        print(">>> Loading Models...")
        
        # 1. Load Grounding DINO (object localization)
        self.dino_processor = AutoProcessor.from_pretrained(DINO_PATH)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_PATH).to(DEVICE)
        
        # 2. Load Qwen2.5-VL (visual reasoning)
        # Optimization: Remove device_map="auto" and accelerate loading, directly use .to(DEVICE)
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_PATH,
            torch_dtype=torch.bfloat16,
            #attn_implementation="flash_attention_2", # Can be deleted if not supported
            device_map="auto"
        )
        self.qwen_processor = AutoProcessor.from_pretrained(QWEN_PATH)
        print(">>> Models Loaded Successfully.")

    def detect_and_crop(self, image, text_prompt="main object"):
        """
        Detect and crop objects using Grounding DINO
        """
        inputs = self.dino_processor(images=image, text=text_prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        # ¡¾Fix¡¿: Removed box_threshold and text_threshold parameters to resolve version conflicts
        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[image.size[::-1]]
        )[0]

        crops = []
        final_labels =[]
        
        # Get all predicted boxes, scores and labels
        scores = results['scores'].cpu()
        boxes = results['boxes'].cpu().numpy()
        labels_str = results['labels']

        # ¡¾Fix¡¿: Manually filter results with low confidence (threshold set to 0.35)
        threshold = 0.35
        valid_indices = torch.where(scores > threshold)[0]

        if len(valid_indices) > 0:
            # Find the box with the highest confidence score among those above the threshold
            best_idx = valid_indices[torch.argmax(scores[valid_indices])]
            box = boxes[best_idx]
            label = labels_str[best_idx]
            
            # Cropping logic
            w, h = image.size
            x1, y1, x2, y2 = box
            # Simple boundary protection
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            cropped_img = image.crop((x1, y1, x2, y2))
            crops.append(cropped_img)
            final_labels.append(label)
            print(f"   [Vision] Detected {label} at {box}")
        else:
            print("   [Vision] No object detected, using full image.")
            crops.append(image)
            final_labels.append("object")
            
        return crops, final_labels

    def run_inference(self, image_path, question):
        """
        Execute the full process: detection -> retrieval -> reasoning
        Optimization: Can refer to the vision-reasoner stack for multi-step reasoning
        """
        original_image = Image.open(image_path).convert("RGB")
        
        # Step 1: Object Localization (Grounding)
        # Built-in optimization: Directly use key objects in the question, default to "main subject"
        crops, labels = self.detect_and_crop(original_image, text_prompt="main subject. bird. car. dog.")
        
        # Step 2: Knowledge Retrieval
        # Retrieve knowledge based on detected labels
        knowledge_context = ""
        for label in labels:
            info = SimpleTools.search_knowledge(label)
            knowledge_context += f"Knowledge about {label}: {info}\n"
        
        # Step 3: Build Reasoning Prompt (RAG)
        # Combine original image, cropped image, and knowledge into one prompt
        final_prompt = f"""
        User Question: {question}
        
        Retrieved Knowledge:
        {knowledge_context}
        
        Instruction:
        Use the provided image and knowledge to answer the question. 
        Focus on fine-grained details shown in the cropped regions.
        """
        
        # Prepare input for Qwen
        # Qwen2.5-VL supports multi-image input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": original_image}, # Original image
                    # {"type": "image", "image": crops[0]},      # Cropped detailed image (commented out to save memory)
                    {"type": "text", "text": final_prompt}
                ]
            }
        ]
        
        # Qwen inference
        text = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.qwen_processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.qwen_model.device)
        
        with torch.no_grad():
            generated_ids = self.qwen_model.generate(** inputs, max_new_tokens=512)
            output_text = self.qwen_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            
        # Remove prompt part, only keep answer (Qwen may repeat the prompt during generation)
        # Optimization: Can directly return the result
        response = output_text.split("assistant\n")[-1].strip()
        return response

# --- 4. Test Execution ---
if __name__ == "__main__":
    # Initialize Pipeline
    pipeline = FineGrainedPipeline()
    
    # Test image and question
    test_image = "midterm/bird/test.jpg" # Replace with actual image path
    test_question = "What specific bird species is this?"
    
    if os.path.exists(test_image):
        print(f"\n>>> Processing: {test_image}")
        answer = pipeline.run_inference(test_image, test_question)
        print("\n=== Final Answer ===")
        print(answer)
    else:
        print("Test image not found, please check path.")