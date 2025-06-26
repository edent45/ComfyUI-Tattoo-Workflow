"""Tattoo Optimizer Gradio Interface
This script sets up a Gradio interface for the Tattoo Optimizer application."""

import os
import time
import numpy as np
from PIL import Image
import gradio as gr
from tattoo_optimizer import WorkflowOptimizer, MASK_DIR, OUTPUT_DIR, BASE_PROMPT
from check import run_workflow_once

#------Functions for Gradio Interface------

def display_mask_preview(input_image, mask_data):
    """Create a visualization of the mask overlaid on the image"""
    if input_image is None or mask_data is None:
        return None

    # Load the input image
    if isinstance(input_image, str):
        img = Image.open(input_image).convert('RGBA')
    else:
        img = Image.fromarray(input_image).convert('RGBA')

    # Handle mask data - ImageEditor returns a dictionary format
    if isinstance(mask_data, dict) and "layers" in mask_data:
        # Create a blank canvas (black)
        mask = Image.new('L', img.size, 0)
        
        # Process each drawing layer
        for layer in mask_data["layers"]:
            if "mask" in layer:
                # Load mask layer
                mask_layer = Image.open(layer["mask"]).convert("L")
                # Paste white (255) using the mask layer for transparency
                mask.paste(255, (0, 0), mask=mask_layer)
    elif isinstance(mask_data, str):
        mask = Image.open(mask_data).convert('L')
    else:
        try:
            mask = Image.fromarray(mask_data).convert('L')
        except:
            print("Error processing mask data:", type(mask_data))
            return None

    # Create colored overlay (red for the tattoo area)
    overlay = Image.new('RGBA', img.size, (255, 0, 0, 0))
    mask_array = np.array(mask)
    mask_indices = mask_array > 50  # Threshold

    overlay_array = np.array(overlay)
    overlay_array[mask_indices, 3] = 128  # Set alpha for red overlay

    overlay = Image.fromarray(overlay_array)

    # Composite the images
    result = Image.alpha_composite(img, overlay)

    return np.array(result)

def extract_background_from_mask_editor(mask_editor_data):
    """Extract the background image path from the mask editor component"""
    if mask_editor_data is None:
        return None
        
    if isinstance(mask_editor_data, dict) and "background" in mask_editor_data:
        background = mask_editor_data["background"]
        if background:  # If there's actually a background image
            return background
    
    return None

def process_mask(mask_data):
    """Process and save mask data from the mask editor"""
    if not mask_data or "layers" not in mask_data or not mask_data["layers"]:
        return None, "❌ No mask drawn or invalid mask format. Please draw a mask first."

    try:
        # Take the first mask layer (you can later combine if needed)
        layer = mask_data["layers"][0]  # shape: (H, W, 3)
        print("Layer min:", np.min(layer), "max:", np.max(layer), "shape:", layer.shape)
        # Convert RGB to grayscale and then to binary mask
        mask_array = np.array(layer)
        gray = np.mean(mask_array, axis=2).astype(np.uint8)  # average R,G,B to grayscale
        binary = (gray > 50).astype(np.uint8) * 255  # binary threshold

        mask_img = Image.fromarray(binary, mode="L")

        # Save the mask
        os.makedirs(MASK_DIR, exist_ok=True)
        mask_path = os.path.join(MASK_DIR, f"accepted_mask_{int(time.time())}.png").replace("\\", "/")
        mask_img.save(mask_path)
        
        # Debugging: Check if the file exists
        if not os.path.exists(mask_path):
            return None, f"❌ Mask file not found after saving: {mask_path}"

        return mask_path, "✅ Mask accepted and saved! Ready for optimization."

            
    except Exception as e:
        print(f"Error processing mask: {e}")
        return None, f"❌ Error processing mask: {str(e)}"

def extract_input_image(mask_data):
    """Extract the input image from the mask editor"""
    if mask_data is None or mask_data[0] is None:
        return None
    try:
        input_image_array = np.array(mask_data[0])  # mask_data[0] contains the input image
        input_image = Image.fromarray(input_image_array).convert("RGB")
        
        # Save the input image to a temporary file
        temp_image_path = os.path.join(OUTPUT_DIR, f"input_image_{int(time.time())}.png").replace("\\", "/")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        input_image.save(temp_image_path)
        
        return temp_image_path
    except Exception as e:
        print(f"Error extracting input image: {e}")
        return None

def file_to_array(img_path):
    if img_path is None:
        return None
    img = Image.open(img_path).convert("RGB")
    return np.array(img)

def run_optimization(input_img, mask_path, prompt, iterations, variations):
    """Run the optimization process with the given parameters"""
    if not input_img:
        return None, "No input image provided", "", None, [], [], None
    
    if not mask_path:
        return None, "No mask provided. Please create a mask first.", "", None, [], [], None
    
    optimizer = WorkflowOptimizer()
    
    results = optimizer.optimize(
        input_img,
        mask_path,
        prompt,
        int(iterations),
        int(variations)
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

#------Create Gradio Interface------

def create_gui():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        # State variables
        mask_path_state = gr.State(value=None)
        
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Tattoo Workflow Optimizer</h1>")
        
        with gr.Row():
            
            # input image column
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("input"):
                        # Input image upload
                        input_image = gr.Image(type="filepath", label="Upload Input Image",height=512, width=512,interactive=True)
                    with gr.Tab("Mask Editor"):
                        # Mask editor
                        mask_editor = gr.ImageMask(
                            label="Draw mask (white areas = tattoo placement)",
                            brush=gr.Brush(colors=["#ffffff"], color_mode="fixed"),
                            height=512,
                            width=512,
                            image_mode="RGB"
                        )
                    
                # Automatically set mask editor background to input image
                input_image.change(
                    fn=file_to_array,
                    inputs=[input_image],
                    outputs=[mask_editor]
                )
                
                mask_editor.input = input_image  # Set the input image as the background for mask editor
                                
                # Mask controls
                accept_mask_btn = gr.Button("Accept Mask", variant="primary")
                mask_status = gr.Markdown("Draw your mask on the input image and click 'Accept Mask'.")
                
                # Base prompt input
                base_prompt = gr.Textbox(label="Base Prompt (for tattoo design)", value=BASE_PROMPT)
                
                # Optimization parameters
                with gr.Row():
                    iterations = gr.Slider(1, 10, value=3, step=1, label="Iterations")
                    prompt_variations = gr.Slider(1, 5, value=3, step=1, label="Prompt Variations")
                
                # Run optimization button
                optimize_btn = gr.Button("Optimize Workflow", variant="primary")
                
                # debug 
                run_btn = gr.Button("Run Workflow Once", variant="secondary")
                
            with gr.Column(scale=1): # Right column for results
                with gr.Tabs():
                    with gr.TabItem("Best Result"):
                        best_image_display = gr.Image(label="Best Result", height=512, width=512)
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
        
        # Update mask editor when input image changes
        input_image.change(
            fn=lambda img: gr.update(value=img, visible=True),
            inputs=[input_image],
            outputs=[mask_editor]
        )
        
        # Process and accept mask
        accept_mask_btn.click(
            fn=process_mask,
            inputs=[mask_editor],
            outputs=[mask_path_state, mask_status]
        )
        
        # Run optimization
        optimize_btn.click(
            fn=run_optimization,
            inputs=[input_image, mask_path_state, base_prompt, iterations, prompt_variations],
            outputs=[best_image_display, best_score, best_prompt, best_params, gallery, history_list, metrics_display]
        )
        
        """#debug run workflow once
        run_btn.click(
            fn=run_workflow_once,
            inputs=[input_image, mask_path_state, base_prompt],
            outputs=[best_image_display, best_score, best_prompt, best_params, gallery, history_list, metrics_display]
        )
        """
    return demo

if __name__ == "__main__":
    demo = create_gui()
    demo.launch()

