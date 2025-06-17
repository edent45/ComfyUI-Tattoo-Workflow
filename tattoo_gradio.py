import gradio as gr
from PIL import Image
import os
import clip
import torch
from transformers import CLIPModel, CLIPProcessor
import time
import requests

# ---- CLIP Init ----
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ---- CONFIG ----
COMFY_API = "http://127.0.0.1:8188"
WORKFLOW_PATH = "./tattoo_crop_api.json"
INPUT_FOLDER = "./Input_Images"
OUTPUT_DIR = "./Results"
PROMPT_TEXT = ""
N_RUNS = 3

# ---- Functions ----
def load_workflow(path):
    """Load the workflow JSON from the specified path."""
    import json
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_workflow(workflow_json, output_dir, run_id,input_image_path):
    """Run the workflow using the ComfyUI API."""
    # Optionally: set a random seed for reproducibility
    seed = int(time.time()) % 100000
    for node in workflow_json["nodes"]:
        if "seed" in node.get("inputs", {}): # Set the seed for each node
            node["inputs"]["seed"] = seed
        # Set the input image path in the workflow
        if "image" in node.get("inputs", {}): # Set the input image for the node
            node["inputs"]["image"] = input_image_path


    res = requests.post(f"{COMFY_API}/prompt", json=workflow_json)
    prompt_id = res.json()["prompt_id"]

    while True:
        time.sleep(1)
        progress = requests.get(f"{COMFY_API}/history/{prompt_id}").json()
        if progress.get("status") == "succeeded":
            break

    saved_files = []
    for node_output in progress["outputs"].values():
        for img in node_output["images"]:
            img_data = requests.get(f"{COMFY_API}/view?filename={img['filename']}&type=output").content
            img_file = os.path.join(output_dir, f"run_{run_id}.png")
            with open(img_file, 'wb') as f:
                f.write(img_data)
            saved_files.append(img_file)
    return saved_files

def generate_images(input_image_path):
    """Generate images using the specified workflow and input image."""
    workflow = load_workflow(WORKFLOW_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_imgs = []
    for i in range(N_RUNS):
        print(f"Running Workflow {i+1}")
        imgs = run_workflow(workflow, OUTPUT_DIR, i,input_image_path)
        all_imgs.extend(imgs)
    return all_imgs

def rank_by_clip(images, prompt_text):
    """Rank images based on their similarity to the prompt using CLIP."""
    scores = []
    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        inputs = clip_processor(text=[prompt_text], images=image, return_tensors="pt", padding=True).to(device)
        outputs = clip_model(**inputs)
        score = outputs.logits_per_image.item()
        scores.append((img_path, score))
        print(f"{os.path.basename(img_path)}: {score:.4f}")
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0]  # best path


def get_input_images():
    return [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def preview_image(image_name):
    if not image_name:
        return None
    return os.path.join(INPUT_FOLDER, image_name)

def run_workflow(image_name, prompt):
    input_path = os.path.join(INPUT_FOLDER, image_name)
    images = generate_images(input_path)
    best_image = rank_by_clip(images, prompt)
    # Return best image and all images as thumbnails
    thumbs = [Image.open(img).resize((128, 128)) for img in images]
    return best_image, thumbs

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "<h1 style='font-family:Brush Script MT, cursive; color:#A3C1AD;'>Tattoo Generator</h1>",
    )
    with gr.Row():
        with gr.Column(scale=1):
            image_list = gr.Dropdown(
                choices=get_input_images(),
                label="Choose an input image",
                value=get_input_images()[0] if get_input_images() else None,
                interactive=True,
            )
            image_preview = gr.Image(label="Input Preview", interactive=False)
            prompt = gr.Textbox(label="Prompt", value=PROMPT_TEXT, placeholder="Describe your tattoo idea...", lines=1)
            run_btn = gr.Button("Run Tattoo Workflow", variant="primary")
        with gr.Column(scale=2):
            best_image = gr.Image(label="Best Result", interactive=False)
            gallery = gr.Gallery(label="All Generated Images", columns=3, height=200)

    image_list.change(fn=preview_image, inputs=image_list, outputs=image_preview)
    run_btn.click(
        fn=run_workflow,
        inputs=[image_list, prompt],
        outputs=[best_image, gallery]
    )

if __name__ == "__main__":
    demo.launch()