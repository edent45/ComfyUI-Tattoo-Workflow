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
from collections import defaultdict

#------CONSTANTS------
COMFY_API = "http://127.0.0.1:8000/"    #"http://127.0.0.1:8188"
WORKFLOW_PATH = "./tattoo_crop_api.json"
COMFYUI_PROJECT_DIR = r"C:\Users\Admin\Desktop\Uni\Project\Tattoo Workflow\ComfyUI-Tattoo-Workflow"
OUTPUT_DIR = r"C:\Users\Admin\Desktop\Uni\Project\Tattoo Workflow\ComfyUI-Tattoo-Workflow\Results" 
LORA_DIR = r"C:\Users\Admin\Documents\ComfyUI\models\loras"  # Directory with tattoo LoRAs
BASE_PROMPT = "flower tattoo"  # Default base prompt
MASK_DIR = r"C:\Users\Admin\Desktop\Uni\Project\Tattoo Workflow\ComfyUI-Tattoo-Workflow\masks"  # Directory to save masks

#------DEFAULT PARAMETERS------
DEFAULT_PARAMS = {
    "lora1": "sdxl tattoo\\SDXL-tattoo-Lora.safetensors",
    "lora1_model_weight": 0.5,
    "lora1_clip_weight": 0.7,
    "sampler": "dpmpp_2s_ancestral_cfg_pp",
    "scheduler": "karras",
    "steps": 70,
    "denoise": 0.65
}

LORAS = [
    "sdxl tattoo\\real-02.safetensors",
    "sdxl tattoo\\SDXL-tattoo-Lora.safetensors",
    "sdxl tattoo\\TheAlly_Tattoo_Helper..safetensors",
    "ginavalentina-01.safetensors",
    "sd tattoo 1.5\\sleeve_tattoo_v3.safetensors.safetensors",
    "SDXL-Lightning\\sdxl_lightning_4step_lora.safetensors"]

#------CLIP INIT------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#------CLASS DEFINITIONS------
class WorkflowOptimizer:
    def __init__(self):
        # Default parameter ranges
        self.param_ranges = {
            "loras": LORAS,  # List of available LoRAs
            "lora_weights": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "samplers": ["dpmpp_2s_ancestral","ddpm" ,"dpmpp_2s_ancestral_cfg_pp"],
            "schedulers": ["karras", "ddim_uniform", "beta"],
            "steps": [ 40, 50, 60, 70],
            "denoise": [0.55, 0.6, 0.65, 0.7]
        }
        
        self.param_usage = defaultdict(lambda: defaultdict(int))  # param_usage[param][value] = count
        
        # Track best parameters and their scores
        self.best_params = None
        self.best_score = 0
        self.best_image = None
        self.history = []
        
    def get_available_loras(self, model = "sdxl"):
        """Get available LoRA models in the directory."""
        if not os.path.exists(LORA_DIR):
            print(f"Warning: LoRA directory {LORA_DIR} not found. Using defaults.")
            return ["sdxl tattoo/real-02.safetensors", "sdxl tattoo/SDXL-tattoo-Lora.safetensors"]
        
        if model == "sdxl":
            lora_dir = "sdxl tattoo"  # Default directory for SDXL LoRAs
        elif model == "sd":
            lora_dir = "tattoo sd 1.5"
        
        loras = []
        
        for file in os.listdir(lora_dir):
            if file.endswith(".safetensors"):
                loras.append(f"{lora_dir}\\{file}")  # This matches the format ComfyUI expects
                #loras.append(os.path.join(lora_dir, file).replace("\\", "/"))  # Ensure correct path format
        if not loras:
            return ["sdxl tattoo/real-02.safetensors", "sdxl tattoo/SDXL-tattoo-Lora.safetensors"]
        return loras
    
    
    def load_workflow(self):
        """Load the workflow JSON."""
        with open(WORKFLOW_PATH, "r", encoding='utf-8') as f:
            return json.load(f)
    
    def update_workflow_params(self, workflow, params, input_image_path, prompt, mask_path):
        """Update workflow with optimization parameters."""
        # Set input image
        if "1" in workflow and "inputs" in workflow["1"] and "image" in workflow["1"]["inputs"]:
            workflow["1"]["inputs"]["image"] = input_image_path
        
        # set mask
        if "73" in workflow and "inputs" in workflow["73"]:
            workflow["73"]["inputs"]["image_path"] = mask_path
            workflow["73"]["inputs"]["invert"] = False  # Ensure mask is not inverted
            workflow["73"]["inputs"]["threshold"] = 127  # Default threshold for binary mask
            
        
        # Set prompt
        if "7" in workflow and "inputs" in workflow["7"] and "text" in workflow["7"]["inputs"]:
            workflow["7"]["inputs"]["text"] = prompt
        
        # Set KSampler parameters
        if "51" in workflow:
            ksampler = workflow["51"]["inputs"]
            ksampler["seed"] = random.randint(1, 10000000000000)
            ksampler["steps"] = params.get("steps", 70)
            ksampler["cfg"] = 6.5  # Default CFG value
            ksampler["sampler_name"] = params.get("sampler", "dpmpp_2s_ancestral_cfg_pp")
            ksampler["scheduler"] = params.get("scheduler", "karras")
            ksampler["denoise"] = params.get("denoise", 0.64)
        
        # Set LoRA parameters for LoraLoader (node 53)
        if "53" in workflow :
            workflow["53"]["inputs"]["lora_name"] = DEFAULT_PARAMS["lora1"]
            workflow["53"]["inputs"]["strength_model"] = params.get("lora1_model_weight", 0.5)
            workflow["53"]["inputs"]["strength_clip"] = params.get("lora1_clip_weight", 0.7)
        
        return workflow
    
    def run_workflow(self, workflow, output_dir, run_id):
        """Run the workflow using the ComfyUI API."""
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            p = {"prompt": workflow}
            data = json.dumps(p).encode('utf-8')
            res = requests.post(f"{COMFY_API}/prompt", data = data)
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
                    print(f"Error: Failed to get history, status code {res.status_code}")
                    continue
                    
                history = res.json()
                if prompt_id in history:
                    outputs = history[prompt_id].get("outputs", {})
                    completed = history[prompt_id].get("status", {}).get("completed", False)
                    
                    if completed:
                        # Download output images
                        for node_id, node_output in outputs.items():
                            if "images" in node_output :
                                for img in node_output["images"]:
                                    img_filename = img["filename"]
                                    img_subfolder = img.get("subfolder", "")
                                    
                                    # For node 71 (final result)
                                    if node_id == "74":
                                        img_data = requests.get(f"{COMFY_API}/view?filename={img_filename}&subfolder={img_subfolder}").content
                                        output_path = os.path.join(output_dir, f"run_{run_id}_{img_filename}").replace("\\","/")  # Ensure correct path format
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
                    text_emb = torch.nn.functional.normalize(outputs.text_embeds, p=2, dim=-1)
                    img_emb = torch.nn.functional.normalize(outputs.image_embeds, p=2, dim=-1)
                    # Cosine similarity
                    similarity = torch.nn.functional.cosine_similarity(text_emb, img_emb).item()
                    scores.append((img_path, similarity))
            except Exception as e: # Handle image processing errors
                print(f"Error processing {img_path}: {e}")
                scores.append((img_path, 0))
        
        # Sort by similarity score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0] if scores else (None, 0)
    
    @staticmethod
    def crop_to_mask(image: Image.Image, mask_path: str, padding=10) -> Image.Image:
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask)
        coords = np.argwhere(mask_np > 0)
        if coords.size == 0:
            return image  # fallback: no mask, return original
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        # Add padding and clip to image size
        x0 = max(x0 - padding, 0)
        y0 = max(y0 - padding, 0)
        x1 = min(x1 + padding, image.width)
        y1 = min(y1 + padding, image.height)
        return image.crop((x0, y0, x1, y1))
    
    def mask_image(self, image, mask_path):
        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        mask = Image.open(mask_path).convert("L").resize(image.size)
        img_np = np.array(image)
        mask_np = np.array(mask)
        img_np[mask_np < 128] = [127, 127, 127]
        return Image.fromarray(img_np)
    
    
    def rank_images(self, image_paths, mask_path, prompt, target_brightness=0.5):
        """Rank images using improved heuristics and CLIP similarity."""
        if not image_paths:
            return None, 0, {}

        # Adjusted weights
        weights = {
            "clip": 0.5,
            "brightness": 0.05,
            "contrast": 0.05,
            "sharpness": 0.15,
            "edge_quality": 0.1,
            "noise_level": 0.05,
            "tattoo_presence": 0.1
        }

        scores = []

        for img_path in image_paths:
            try:
                pil_img = Image.open(img_path).convert("RGB")
                cropped_img = WorkflowOptimizer.crop_to_mask(pil_img, mask_path)
                masked_img = self.mask_image(pil_img, mask_path)

                # Resize masked_img to match cropped_img for fair comparison
                masked_resized = masked_img.resize(cropped_img.size)
                masked_gray = cv2.cvtColor(np.array(masked_resized), cv2.COLOR_RGB2GRAY)

                cv_img = np.array(cropped_img)
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

                # ----- CLIP Similarity -----
                improved_prompt = prompt + " on skin"
                inputs_1 = clip_processor(text=[improved_prompt], images=cropped_img, return_tensors="pt", padding=True).to(device)
                inputs_2 = clip_processor(text=[prompt], images=masked_resized, return_tensors="pt", padding=True).to(device)

                similarities = []
                for inputs in [inputs_1, inputs_2]:
                    with torch.no_grad():
                        outputs = clip_model(**inputs)
                        text_emb = torch.nn.functional.normalize(outputs.text_embeds, p=2, dim=-1)
                        img_emb = torch.nn.functional.normalize(outputs.image_embeds, p=2, dim=-1)
                        sim = torch.nn.functional.cosine_similarity(text_emb, img_emb).item()
                        similarities.append(sim)
                similarity = max(similarities)
                similarity_score = min(similarity * 1.5, 1.0)  # Boosted

                # ----- Tattoo Presence (edge density in mask) -----
                edges = feature.canny(gray, sigma=2)
                tattoo_presence_mask = masked_gray > 64
                tattoo_area = edges[tattoo_presence_mask]
                if tattoo_area.size > 0:
                    tattoo_presence = np.clip(np.mean(tattoo_area) * 10.0, 0.0, 1.0)
                else:
                    tattoo_presence = 0.1  # fallback small value

                # ----- Brightness -----
                brightness = np.mean(gray) / 255.0
                brightness_score = 1.0 - abs(brightness - target_brightness)

                # ----- Contrast -----
                contrast = np.std(gray) / 255.0
                contrast_score = min(contrast * 2.5, 1.0)

                # ----- Sharpness -----
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                sharpness = np.var(laplacian)
                sharpness_score = np.clip(sharpness / 300.0, 0.0, 1.0)

                # ----- Edge Quality -----
                edge_quality = np.clip(np.mean(edges) * 3.0, 0.0, 1.0)

                # ----- Noise Level -----
                blurred = cv2.GaussianBlur(gray, (11, 11), 0)
                noise = np.mean(np.abs(gray.astype(np.float32) - blurred.astype(np.float32)))
                noise_score = 1.0 - min(noise / 20.0, 1.0)

                # ----- Final Weighted Score -----
                combined_score = (
                    weights["clip"] * similarity_score +
                    weights["brightness"] * brightness_score +
                    weights["contrast"] * contrast_score +
                    weights["sharpness"] * sharpness_score +
                    weights["edge_quality"] * edge_quality +
                    weights["noise_level"] * noise_score +
                    weights["tattoo_presence"] * tattoo_presence
                )

                # ----- Final Adjustment -----
                combined_score = min(combined_score * 1.2 + 0.1, 1.0)  # Rescale

                # ----- Debug print -----
                print(f"[{img_path}] Score = {combined_score:.3f} | similarity={similarity_score:.3f} | sharpness={sharpness_score:.3f} | tattoo={tattoo_presence:.3f}")

                metrics = {
                    "similarity": similarity_score,
                    "brightness": brightness_score,
                    "contrast": contrast_score,
                    "sharpness": sharpness_score,
                    "edge_quality": edge_quality,
                    "noise_level": noise_score,
                    "tattoo_presence": tattoo_presence,
                    "combined_score": combined_score
                }

                scores.append((img_path, combined_score, metrics))

            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                scores.append((img_path, 0, {}))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0] if scores else (None, 0, {})
    
    
    def generate_prompt_variations(self, base_prompt, n=3):
        """Generate variations of the base prompt."""
        # List of tattoo style descriptors
        style_descriptors = [
            "realistic", "watercolor", "traditional", "japanese",
            "black and grey", "old looking tattoo", "minimalist", 
            "linework", "illustrative", "fine line"
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
        
        # When selecting a value:
        for param, choices in self.param_ranges.items():
            if param == "denoise":
                continue  # Skip denoise for now, handled separately
            available = [v for v in choices if self.param_usage[param][v] < 4]
            if not available:
                available = choices  # fallback if all are used up
            value = random.choice(available)
            params[param] = value
            self.param_usage[param][value] += 1
            
        params["denoise"] = round(random.uniform(0.55, 0.7), 2)
        
        """# Random KSampler parameters
        params["steps"] = random.choice(self.param_ranges["steps"]) 
        params["sampler"] = random.choice(self.param_ranges["samplers"])
        params["scheduler"] = random.choice(self.param_ranges["schedulers"])
        params["denoise"] = round(random.uniform(0.55, 0.7), 2)
        
        # Random LoRA parameters
        available_loras = self.param_ranges["loras"]
        if available_loras:
            # LoraLoader parameters (for node 53)
            params["lora1"] = random.choice(available_loras)
            params["lora1_model_weight"] = random.choice(self.param_ranges["lora_weights"])
            params["lora1_clip_weight"] = random.choice(self.param_ranges["lora_weights"])"""
        
        return params
    
    def optimize(self, input_image_path,mask_path, base_prompt, iterations=5, prompt_variations=3):
        """Run optimization process with multiple iterations."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        all_results = []
        bad_params = set()  # Store hashes of bad parameter sets
        
        if base_prompt is None or base_prompt.strip() == "":
            raise ValueError("Base prompt cannot be empty")
        
        # Generate prompt variations
        prompts = self.generate_prompt_variations(base_prompt, prompt_variations)
        
        for iteration in range(iterations):
            score_threshold = 0.2 if iteration == 0 else 0.4  # Adjust threshold based on iteration
            for prompt_idx, prompt in enumerate(prompts):
                print(f"Iteration {iteration+1}/{iterations}, Prompt {prompt_idx+1}/{len(prompts)}")
                print(f"Prompt: {prompt}")
                
                # Use default params for the first run, random for others
                if iteration == 0 and prompt_idx == 0:
                    params = DEFAULT_PARAMS.copy()
                else:
                    # Try up to 10 times to get a "new" parameter set
                    for _ in range(10):
                        params = self.generate_random_params()
                        params_hash = str(sorted(params.items()))
                        if params_hash not in bad_params:
                            break

                print(f"Parameters: {params}")
                
                # Update workflow with params
                workflow = self.load_workflow()
                workflow = self.update_workflow_params(workflow, params, input_image_path, prompt,mask_path)
                
                # Run workflow
                run_id = f"{iteration}_{prompt_idx}"
                output_images = self.run_workflow(workflow, OUTPUT_DIR, run_id)
                
                if output_images:
                    # Rank by CLIP and other metrics
                    best_img_path, score, metrics = self.rank_images(output_images,mask_path, prompt)
                    
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
                        
                        if score < score_threshold:
                            # Mark these params as "bad"
                            if not (params.get("sampler") == "ddpm" or abs(params.get("denoise", 0) - 0.65) < 1e-6):
                                params_hash = str(sorted(params.items()))
                                bad_params.add(params_hash)
                                print(f"Low score ({score:.4f}), will not reuse these parameters.")
                            else:
                                print(f"Low score ({score:.4f}), but keeping parameters due to exception (sampler=ddpm or denoise=0.65).")
        
        # Sort results by score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results
