""" Tattoo Workflow Optimizer
This module provides a class to optimize tattoo workflows by analyzing and adjusting the weights or LORA models used in the process.
The optimizer can be used to enhance the quality of tattoo designs by fine-tuning the parameters.
"""

#------IMPORTS------
import os
import time
import json
import random
import requests
import torch
import numpy as np
import cv2
from skimage import metrics, exposure, feature
from PIL import Image
import gradio as gr
from transformers import CLIPProcessor, CLIPModel

#------CONSTANTS------
COMFY_API = "http://127.0.0.1:8000/"    #"http://127.0.0.1:8188"
WORKFLOW_PATH = "./tattoo_crop_api.json"
OUTPUT_DIR = "./output_images"
LORA_DIR = r"C:\Users\Admin\Documents\ComfyUI\models\loras"  # Directory with tattoo LoRAs
BASE_PROMPT = "flower tattoo"  # Default base prompt

#------DEFAULT PARAMETERS------
DEFAULT_PARAMS = {
    "lora1": "tattoo/sleeve_tattoo_v3.safetensors",
    "lora1_model_weight": 0.7,
    "lora1_clip_weight": 0.85,
    "sampler": "dpmpp_2s_ancestral_cfg_pp",
    "scheduler": "karras",
    "steps": 70,
    "cfg": 6.5,
    "denoise": 0.64
}

#------CLIP INIT------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#------CLASS DEFINITIONS------
class WorkflowOptimizer:
    def __init__(self):
        # Default parameter ranges
        self.param_ranges = {
            "loras": self.get_available_loras(),
            "lora_weights": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "samplers": ["dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_2m", "euler", "euler_ancestral"],
            "schedulers": ["karras", "exponential", "normal"],
            "steps": [20, 30, 40, 50, 60, 70],
            "cfg": [5.0, 6.0, 7.0, 8.0],
            "denoise": [0.5, 0.6, 0.7, 0.8]
        }
        
        # Track best parameters and their scores
        self.best_params = None
        self.best_score = 0
        self.best_image = None
        self.history = []
        
    def get_available_loras(self):
        """Get available LoRA models in the directory."""
        if not os.path.exists(LORA_DIR):
            print(f"Warning: LoRA directory {LORA_DIR} not found. Using defaults.")
            return ["tattoo/sleeve_tattoo_v3.safetensors", "tattoo_lora01.safetensors"]
            
        loras = []
        for file in os.listdir(LORA_DIR):
            if file.endswith(".safetensors"):
                loras.append(os.path.join("tattoo", file))
        if not loras:
            return ["tattoo/sleeve_tattoo_v3.safetensors", "tattoo_lora01.safetensors"]
        return loras
    
    def load_workflow(self):
        """Load the workflow JSON."""
        with open(WORKFLOW_PATH, "r" , encoding='utf-8') as f:
            return json.load(f)
    
    def update_workflow_params(self, workflow, params, input_image_path, prompt):
        """Update workflow with optimization parameters."""
        # Set input image
        if "1" in workflow and "inputs" in workflow["1"] and "image" in workflow["1"]["inputs"]:
            workflow["1"]["inputs"]["image"] = input_image_path
        
        # Set prompt
        if "7" in workflow and "inputs" in workflow["7"] and "text" in workflow["7"]["inputs"]:
            workflow["7"]["inputs"]["text"] = prompt
        
        # Set KSampler parameters
        if "51" in workflow:
            ksampler = workflow["51"]["inputs"]
            ksampler["seed"] = random.randint(1, 10000000000000)
            ksampler["steps"] = params.get("steps", 70)
            ksampler["cfg"] = params.get("cfg", 6.5)
            ksampler["sampler_name"] = params.get("sampler", "dpmpp_2s_ancestral_cfg_pp")
            ksampler["scheduler"] = params.get("scheduler", "karras")
            ksampler["denoise"] = params.get("denoise", 0.64)
        
        # Set LoRA parameters for LoraLoader (node 53)
        if "53" in workflow and params.get("lora1"):
            lora_loader = workflow["53"]["inputs"]
            lora_loader["lora_name"] = params["lora1"]
            lora_loader["strength_model"] = params.get("lora1_model_weight", 0.7)
            lora_loader["strength_clip"] = params.get("lora1_clip_weight", 0.85)
        
        return workflow
    
    def run_workflow(self, workflow, output_dir, run_id):
        """Run the workflow using the ComfyUI API."""
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            res = requests.post(f"{COMFY_API}/prompt", json=workflow)
            if res.status_code != 200:
                print(f"Error: API returned status code {res.status_code}")
                return []
                
            data = res.json()
            prompt_id = data.get("prompt_id")
            
            if not prompt_id:
                print("Error: No prompt_id returned from API")
                return []
                
            # Wait for execution to complete
            completed = False
            output_images = []
            
            while not completed:
                time.sleep(1)
                res = requests.get(f"{COMFY_API}/history")
                if res.status_code != 200:
                    continue
                    
                history = res.json()
                if prompt_id in history:
                    outputs = history[prompt_id].get("outputs", {})
                    completed = history[prompt_id].get("status", {}).get("completed", False)
                    
                    if completed:
                        # Download output images
                        for node_id, node_output in outputs.items():
                            if "images" in node_output:
                                for img in node_output["images"]:
                                    img_filename = img["filename"]
                                    img_subfolder = img.get("subfolder", "")
                                    
                                    # For node 71 (final result)
                                    if node_id == "71":
                                        img_data = requests.get(f"{COMFY_API}/view?filename={img_filename}&subfolder={img_subfolder}").content
                                        output_path = os.path.join(output_dir, f"run_{run_id}_{img_filename}")
                                        with open(output_path, "wb") as f:
                                            f.write(img_data)
                                        output_images.append(output_path)
            
            return output_images
            
        except Exception as e:
            print(f"Error running workflow: {e}")
            return []
    
    def rank_by_clip(self, image_paths, prompt):
        """Rank images by CLIP similarity to prompt."""
        if not image_paths:
            return None, 0
            
        scores = []
        
        for img_path in image_paths:
            try:
                # Load and process image 
                img = Image.open(img_path).convert("RGB")
                inputs = clip_processor(text=[prompt], images=img, return_tensors="pt", padding=True).to(device) # Process text and image inputs
                
                with torch.no_grad(): 
                    outputs = clip_model(**inputs) # Get CLIP embeddings
                    text_emb = outputs.text_embeds # Get text embeddings
                    img_emb = outputs.image_embeds # Get image embeddings
                    
                    # Cosine similarity
                    similarity = torch.nn.functional.cosine_similarity(text_emb, img_emb).item()
                    scores.append((img_path, similarity))
            except Exception as e: # Handle image processing errors
                print(f"Error processing {img_path}: {e}")
                scores.append((img_path, 0))
        
        # Sort by similarity score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0] if scores else (None, 0)
    
    def rank_images(self, image_paths, prompt, target_brightness=0.5, style_reference=None):
        """Rank images using multiple criteria including CLIP similarity, brightness, contrast, etc."""
        if not image_paths:
            return None, 0
        
        # weights
        weights = {
            "clip": 0.7,
            "brightness": 0.1,
            "contrast": 0.1,
            "sharpness": 0.1,
            "edge_quality": 0.1,
            "noise_level": 0.05
        }
    
        scores = []
        for img_path in image_paths:
            # Load and process image
            try:
                # Open image with PIL for CLIP
                pil_img = Image.open(img_path).convert("RGB")
                
                #convert opencv image
                cv_img = np.array(pil_img)
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                
                # clip similarity
                inputs = clip_processor(text=[prompt], images=pil_img, return_tensors="pt", padding=True).to(device) # Process text and image inputs
                
                with torch.no_grad(): 
                    outputs = clip_model(**inputs) # Get CLIP embeddings
                    text_emb = outputs.text_embeds # Get text embeddings
                    img_emb = outputs.image_embeds # Get image embeddings
                    # Cosine similarity
                    similarity = torch.nn.functional.cosine_similarity(text_emb, img_emb).item()
                
                # brightness
                brightness = np.mean(gray) / 255.0  # Normalize to [0, 1]
                brightness_score = 1.0 - abs(brightness - target_brightness)   #  # Higher when close to target
                
                # contrast
                contrast = np.std(gray) / 255.0  # Normalize to [0, 1]
                contrast_score = min(contrast * 2.5, 1.0)  # Scale to [0, 1]
                
                # sharpness
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                sharpness = np.var(laplacian)
                sharpness_score = min(sharpness / 500.0, 1.0)  # Normalize and cap
                
                #edge quality
                edges = feature.canny(gray, sigma=2)
                edge_quality = np.mean(edges) * 5.0  # Scale up edge detection result
                edge_quality = min(edge_quality, 1.0)  # Cap at 1.0
                
                #noise level - low is better
                blurred = cv2.GaussianBlur(gray, (11, 11), 0)
                noise = np.mean(np.abs(gray.astype(np.float32) - blurred.astype(np.float32)))
                noise_score = 1.0 - min(noise / 20.0, 1.0)  # Invert and normalize
                
                #weighted score
                combined_score = (
                    weights["clip"] * similarity +
                    weights["brightness"] * brightness_score +
                    weights["contrast"] * contrast_score +
                    weights["sharpness"] * sharpness_score +
                    weights["edge_quality"] * edge_quality +
                    weights["noise_level"] * noise_score
                )
                
                #metrics 
                metrics = {
                    "similarity": similarity,
                    "brightness": brightness_score,
                    "contrast": contrast_score,
                    "sharpness": sharpness_score,
                    "edge_quality": edge_quality,
                    "noise_level": noise_score,
                    "combined_score": combined_score
                }
                
                scores.append((img_path, combined_score, metrics))
            
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                scores.append((img_path, 0, {}))
            
        # Sort by combined score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0] if scores else (None, 0, {})
    
    def generate_prompt_variations(self, base_prompt, n=3):
        """Generate variations of the base prompt."""
        # List of tattoo style descriptors
        style_descriptors = [
            "realistic", "watercolor", "traditional", "japanese", "tribal",
            "black and grey", "neo-traditional", "minimalist", "sketch", "dotwork",
            "linework", "illustrative", "blackwork", "fine line", "surreal"
        ]
        
        # List of quality enhancers
        quality_enhancers = [
            "detailed", "high contrast", "intricate", "elegant", "flowing",
            "beautiful", "artistic", "professional", "masterpiece", "high quality",
            "balanced composition", "photorealistic", "award winning", "sharp details"
        ]
        
        variations = [base_prompt]  # Include original
        
        # Create prompt variations by adding style and quality terms
        for _ in range(n-1):
            style = random.choice(style_descriptors)
            quality = random.choice(quality_enhancers)
            variation = f"{base_prompt}, {style} style, {quality}"
            variations.append(variation)
            
        return variations
    
    def generate_random_params(self):
        """Generate random parameters for optimization."""
        params = {}
        
        # Random KSampler parameters
        params["steps"] = random.choice(self.param_ranges["steps"])
        params["cfg"] = random.choice(self.param_ranges["cfg"])
        params["sampler"] = random.choice(self.param_ranges["samplers"])
        params["scheduler"] = random.choice(self.param_ranges["schedulers"])
        params["denoise"] = random.choice(self.param_ranges["denoise"])
        
        # Random LoRA parameters
        available_loras = self.param_ranges["loras"]
        if available_loras:
            # LoraLoader parameters (for node 53)
            params["lora1"] = random.choice(available_loras)
            params["lora1_model_weight"] = random.choice(self.param_ranges["lora_weights"])
            params["lora1_clip_weight"] = random.choice(self.param_ranges["lora_weights"])
        
        return params
    
    def optimize(self, input_image_path, base_prompt, iterations=5, prompt_variations=3):
        """Run optimization process with multiple iterations."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        all_results = []
        
        # Generate prompt variations
        prompts = self.generate_prompt_variations(base_prompt, prompt_variations)
        
        for iteration in range(iterations):
            for prompt_idx, prompt in enumerate(prompts):
                print(f"Iteration {iteration+1}/{iterations}, Prompt {prompt_idx+1}/{len(prompts)}")
                print(f"Prompt: {prompt}")
                
                # Generate random parameters
                params = self.generate_random_params()
                print(f"Parameters: {params}")
                
                # Update workflow with params
                workflow = self.load_workflow()
                workflow = self.update_workflow_params(workflow, params, input_image_path, prompt)
                
                # Run workflow
                run_id = f"{iteration}_{prompt_idx}"
                output_images = self.run_workflow(workflow, OUTPUT_DIR, run_id)
                
                if output_images:
                    # Rank by CLIP and other metrics
                    best_img_path, score, metrics = self.rank_images(output_images, prompt)
                    
                    if best_img_path:
                        result = {
                            "image_path": best_img_path,
                            "prompt": prompt,
                            "params": params,
                            "score": score,
                            "metrics": metrics
                        }
                        all_results.append(result)
                        
                        # Update best result if better
                        if score > self.best_score:
                            self.best_score = score
                            self.best_params = params.copy()
                            self.best_params["prompt"] = prompt
                            self.best_image = best_img_path
                            print(f"New best result! Score: {score:.4f}")
                            print(f"Image: {best_img_path}")
                        
                        # Add to history
                        self.history.append(result)
        
        # Sort results by score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results

# GUI for optimization
def create_gui():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        optimizer = WorkflowOptimizer()
        
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Tattoo Workflow Optimizer</h1>")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="filepath", label="Input Image")
                
                with gr.Group():
                    base_prompt = gr.Textbox(label="Base Prompt", value=BASE_PROMPT)
                    iterations = gr.Slider(1, 10, value=3, step=1, label="Iterations")
                    prompt_variations = gr.Slider(1, 5, value=3, step=1, label="Prompt Variations")
                
                optimize_btn = gr.Button("Optimize Workflow", variant="primary")
                
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Best Result"):
                        best_image_display = gr.Image(label="Best Result")
                        best_score = gr.Textbox(label="Score", interactive=False)
                        best_prompt = gr.Textbox(label="Prompt", interactive=False)
                        
                        with gr.Accordion("Best Parameters", open=False):
                            best_params = gr.JSON(label="Parameters")
                    
                    with gr.TabItem("All Results"):
                        gallery = gr.Gallery(label="Results Gallery", columns=3, height=600)
                        
                    with gr.TabItem("History"):
                        history_list = gr.Dataframe(
                            headers=["Iteration", "Score", "Prompt"],
                            label="Optimization History"
                        )
                    
                    with gr.TabItem("Metrics"):
                        metrics_display = gr.JSON(label="Image Metrics")
        
        def run_optimization(input_img, prompt, iters, variations):
            if not input_img:
                return None, "No input image selected", "", None, [], [], None
                
            results = optimizer.optimize(
                input_img, 
                prompt, 
                iterations=int(iters), 
                prompt_variations=int(variations)
            )
            
            if not results:
                return None, "No results generated", "", None, [], [], None
            
            # Format for gallery
            gallery_images = [(r["image_path"], f"Score: {r['score']:.4f}") for r in results]
            
            # Format for history
            history_data = []
            for i, r in enumerate(optimizer.history):
                history_data.append([i+1, f"{r['score']:.4f}", r['prompt']])
            
            # Get metrics for best result
            best_metrics = results[0].get("metrics", {})
            
            return (
                optimizer.best_image,
                f"{optimizer.best_score:.4f}",
                optimizer.best_params.get("prompt", ""),
                optimizer.best_params,
                gallery_images,
                history_data,
                best_metrics
            )
        
        optimize_btn.click(
            run_optimization,
            inputs=[input_image, base_prompt, iterations, prompt_variations],
            outputs=[best_image_display, best_score, best_prompt, best_params, gallery, history_list, metrics_display]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_gui()
    demo.launch()