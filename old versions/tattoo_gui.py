import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageFilter
import os
import time
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import ttk


# ---- CLIP Init ----
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ---- CONFIG ----
COMFY_API = "http://127.0.0.1:8188"
WORKFLOW_PATH = "./tattoo_crop_api.json"
OUTPUT_DIR = "./Results"
PROMPT_TEXT = "flower tattoo"
N_RUNS = 3
BACKGROUND_COLOR = "#B6B6E4"  # Soft pastel background

# ---- Functions ----
def load_workflow(path):
    """Load the workflow JSON from the specified path."""
    import json
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_workflow(workflow_json, output_dir, run_id,input_image_path,prompt_text):
    """Run the workflow using the ComfyUI API."""
    # Optionally: set a random seed for reproducibility
    seed = int(time.time()) % 100000
    """for node in workflow_json["nodes"]:
        if "seed" in node.get("inputs", {}): # Set the seed for each node
            node["inputs"]["seed"] = seed
        # Set the input image path in the workflow
        if "image" in node.get("inputs", {}): # Set the input image for the node
            node["inputs"]["image"] = input_image_path

    
    res = requests.post(f"{COMFY_API}/prompt", json=workflow_json)"""
    
    # For workflow with numbered node keys (not "nodes" array)
    for node_id, node in workflow_json.items():
        if "inputs" in node and "seed" in node["inputs"]:
            node["inputs"]["seed"] = seed
        # Update node 1 (LoadImage) with our selected image
        if node_id == "1" and "inputs" in node and "image" in node["inputs"]:
            node["inputs"]["image"] = input_image_path
        # To update the prompt 
        if node_id == "7" and "inputs" in node and "text" in node["inputs"]:
            node["inputs"]["text"] = prompt_text  # You'll need to pass prompt_text to the function
    
    # Continue with API request
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

def generate_images(input_image_path, prompt_text):
    """Generate images using the specified workflow , input image , and prompt text."""
    workflow = load_workflow(WORKFLOW_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_imgs = []
    for i in range(N_RUNS):
        print(f"Running Workflow {i+1}")
        imgs = run_workflow(workflow, OUTPUT_DIR, i, input_image_path, prompt_text)
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

# ---- GUI ----

class TattooApp:
    """Tattoo Generator GUI Application."""
    def __init__(self, root):
        self.root = root
        self.root.title("Tattoo Generator")
        self.root.geometry("800x1000")
        self.root.resizable(True, True)
        self.root.configure(bg="#F6F5F3")  # Soft pastel background

        # Input images folder and files (define first!)
        self.input_folder = "./Input_Images"
        self.image_files = [f for f in os.listdir(self.input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.input_image_idx = 0

        # Pastel accent colors
        pastel_accent = "#A3C1AD"
        pastel_button = "#F7CAC9"
        pastel_frame = "#E3E6F3"
        pastel_text = "#6B5B95"

        # Logo/banner
        banner_img = None
        banner_path = os.path.join(os.path.dirname(__file__), "logo.png")
        if os.path.exists(banner_path):
            banner_img = Image.open(banner_path).resize((100, 100))
            banner_img = ImageTk.PhotoImage(banner_img)
            self.banner = tk.Label(root, image=banner_img, bg="#F6F5F3")
            self.banner.image = banner_img
            self.banner.pack(pady=(20, 5))
        # Loopy font only for banner
        self.title_label = tk.Label(root, text="Tattoo Generator", font=("Brush Script MT", 30, "bold"), fg="#A3C1AD", bg="#F6F5F3")
        self.title_label = tb.Label(root, text="Tattoo Generator", font=("Brush Script MT", 30, "bold"), foreground=pastel_accent, background="#F6F5F3")
        self.title_label.pack(pady=(0, 18))
        # Add a label for showing the current image filename (now hidden)
        self.image_name_label = tk.Label(root, text="", font=("Arial", 12), fg="#6B5B95", bg="#F6F5F3")
        self.image_name_label.pack_forget()

        # Main container with two sides
        main_frame = tk.Frame(root, bg="#F6F5F3")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        left_frame = tk.Frame(main_frame, bg="#F6F5F3")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        right_frame = tk.Frame(main_frame, bg="#F6F5F3")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        

        # Left side: Input carousel (top half)
        input_area = tk.Frame(left_frame, bg="#E3E6F3", bd=2, relief=tk.RIDGE, height=350)
        input_area.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        input_area.pack_propagate(False)
        self.carousel_frame = tk.Frame(input_area, bg="#E3E6F3")
        self.carousel_frame.pack(expand=True)
        self.left_img_label = tk.Label(self.carousel_frame, bg="#E3E6F3")
        self.left_img_label.pack(side=tk.LEFT, padx=10)
        self.center_img_label = tk.Label(self.carousel_frame, bg="#E3E6F3")
        self.center_img_label.pack(side=tk.LEFT, padx=10)
        self.right_img_label = tk.Label(self.carousel_frame, bg="#E3E6F3")
        self.right_img_label.pack(side=tk.LEFT, padx=10)
        nav_frame = tk.Frame(input_area, bg="#E3E6F3")
        nav_frame.pack(pady=8)
        icons_folder = os.path.join(os.path.dirname(__file__), "ICONS")
        prev_path = os.path.join(icons_folder, "angle-left.png")
        next_path = os.path.join(icons_folder, "angle-right.png")
        prev_icon = ImageTk.PhotoImage(Image.open(prev_path).resize((16, 16))) if os.path.exists(prev_path) else None
        next_icon = ImageTk.PhotoImage(Image.open(next_path).resize((16, 16))) if os.path.exists(next_path) else None
        self.prev_btn = tb.Button(nav_frame, text="", image=prev_icon, command=self.show_prev_input_image, bootstyle="secondary", cursor="hand2", width=2)
        self.prev_btn.image = prev_icon
        self.prev_btn.pack(side=tk.LEFT, padx=8)
        self.select_btn = tb.Button(nav_frame, text="Select This Image", command=self.select_current_input_image, bootstyle="outline-toolbutton", cursor="hand2", width=24)
        self.select_btn.pack(side=tk.LEFT, padx=18, ipadx=10, ipady=8)
        self.next_btn = tb.Button(nav_frame, text="", image=next_icon, command=self.show_next_input_image, bootstyle="secondary", cursor="hand2", width=2)
        self.next_btn.image = next_icon
        self.next_btn.pack(side=tk.LEFT, padx=8)
        self.selected_image = tk.StringVar()
        if self.image_files:
            self.selected_image.set(self.image_files[0])
            self.show_input_image()
            
        # Prompt text area
        left_bottom_frame = tk.Frame(left_frame, bg="#E3E6F3")
        left_bottom_frame.pack(fill=tk.X, pady=(10, 0))
        left_bottom_frame.pack_propagate(False)
        left_bottom_frame.config(height=100)  # Set a fixed height for the prompt area
        left_bottom_frame.pack_propagate(False)
        prompt_label = tk.Label(
            left_bottom_frame,  
            text="Prompt:",
            font=("Segoe Script", 14, "bold"),   # Loopy, stylish font
            fg="#6B5B95",                        # Pastel purple text
            bg="#F6F5F3",                        # Soft pastel background
            padx=12, pady=6,                     # Padding
            relief="groove",                     # Optional: adds a border
            bd=2,                                # Optional: border width
            cursor="xterm"                       # Optional: text cursor on hover
        )
        prompt_label.pack(pady=(12, 4))
        self.prompt_var = tk.StringVar(value=PROMPT_TEXT)
        self.prompt_entry = tk.Entry(left_bottom_frame, textvariable=self.prompt_var, font=("Arial", 12), width=40)
        self.prompt_entry.pack(pady=(0, 10))

        # Left side: Results area (bottom half)
        results_area = tk.Frame(left_frame, bg=pastel_frame, bd=2, relief=tk.RIDGE, height=350)
        results_area.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        results_area.pack_propagate(False)
        self.thumb_frame = tk.Frame(results_area, bg=pastel_frame)
        self.thumb_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.thumb_labels = []

        # Right side: Input and best image with slider
        compare_area = tk.Frame(right_frame, bg="#F6F5F3", bd=2, relief=tk.RIDGE)
        compare_area.pack(fill=tk.BOTH, expand=True)
        compare_area.pack_propagate(False)
        self.compare_canvas = tk.Canvas(compare_area, width=400, height=400, bg=pastel_frame, highlightthickness=0)
        self.compare_canvas.pack(pady=30)
        self.slider = ttk.Scale(compare_area, from_=0, to=1, orient=tk.HORIZONTAL, value=0, command=self.update_compare_slider)
        self.slider.pack(fill=tk.X, padx=40, pady=10)
        self.input_compare_img = None
        self.best_compare_img = None
        self.compare_img_tk = None

        # Progress bar and buttons below main_frame
        self.progress = tb.Progressbar(root,mode='indeterminate', length=350, bootstyle="info")
        self.progress.pack(pady=18)
        self.progress.stop()
        self.progress.pack_forget()
        self.generate_btn = tb.Button(root, text="Run Tattoo Workflow", command=self.run, bootstyle="info", cursor="hand2")
        self.generate_btn.pack(pady=10)
        self.show_all_btn = tb.Button(root, text="Show All Images", command=self.show_all_images, state=tk.DISABLED, bootstyle="primary", cursor="hand2")
        self.show_all_btn.pack(pady=5)
        self.result_label = tk.Label(root, text="", font=("Arial", 16), fg=pastel_text, bg="#F6F5F3")
        self.result_label.pack(pady=10)

    def show_input_image(self):
        if not self.image_files:
            return
        n = len(self.image_files)
        idx = self.input_image_idx
        prev_idx = (idx - 1) % n
        next_idx = (idx + 1) % n
        # Center image (sharp, contained pop effect)
        center_path = os.path.join(self.input_folder, self.image_files[idx])
        center_img = Image.open(center_path)
        center_img.thumbnail((220, 220), Image.LANCZOS)
        center_img_tk = ImageTk.PhotoImage(center_img)
        # Left image (blurred)
        left_path = os.path.join(self.input_folder, self.image_files[prev_idx])
        left_img = Image.open(left_path)
        left_img.thumbnail((120, 120), Image.LANCZOS)
        left_img = left_img.filter(ImageFilter.BLUR)
        left_img_tk = ImageTk.PhotoImage(left_img)
        # Right image (blurred)
        right_path = os.path.join(self.input_folder, self.image_files[next_idx])
        right_img = Image.open(right_path)
        right_img.thumbnail((120, 120), Image.LANCZOS)
        right_img = right_img.filter(ImageFilter.BLUR)
        right_img_tk = ImageTk.PhotoImage(right_img)
        # Update labels
        self.left_img_label.configure(image=left_img_tk)
        self.left_img_label.image = left_img_tk
        self.right_img_label.configure(image=right_img_tk)
        self.right_img_label.image = right_img_tk
        # Pop effect only for center image label
        def pop_anim(step=0):
            sizes = [220, 250, 220]
            if step < len(sizes):
                img = Image.open(center_path)
                img.thumbnail((sizes[step], sizes[step]), Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)
                self.center_img_label.configure(image=img_tk)
                self.center_img_label.image = img_tk
                self.root.after(60, lambda: pop_anim(step+1))
            else:
                self.center_img_label.configure(image=center_img_tk)
                self.center_img_label.image = center_img_tk
        pop_anim(0)
        # Do not show filename

    def show_prev_input_image(self):
        if not self.image_files:
            return
        self.input_image_idx = (self.input_image_idx - 1) % len(self.image_files)
        self.show_input_image()

    def show_next_input_image(self):
        if not self.image_files:
            return
        self.input_image_idx = (self.input_image_idx + 1) % len(self.image_files)
        self.show_input_image()

    def select_current_input_image(self):
        if not self.image_files:
            return
        self.selected_image.set(self.image_files[self.input_image_idx])
        # Do not show filename

    def run(self):
        """Run the tattoo generation workflow with the selected input image."""
        
        chosen_image_path = os.path.join(self.input_folder, self.selected_image.get()) # Get the selected image path
        print(f"Selected input image: {chosen_image_path}")
        prompt_text = self.prompt_var.get() # Get the prompt text from the entry field
        self.result_label.config(text="Running Workflow...") # Show initial message
        self.progress.pack(pady=18) # Show the progress bar
        self.progress.start()
        self.root.update_idletasks()
        
        if not prompt_text: # If no prompt text is provided, use the default
            prompt_text = PROMPT_TEXT
        
        # Generate images using the workflow
        self.generated_images = generate_images(chosen_image_path, prompt_text) # Generate images using the workflow
        self.current_image_idx = 0 # Reset current image index
        self.show_all_btn.config(state=tk.NORMAL) # Enable the "Show All Images" button

        best_image = rank_by_clip(self.generated_images, prompt_text) # Rank images using CLIP
        
        # For right side comparison slider
        self.input_compare_img = Image.open(chosen_image_path).resize((400, 400), Image.LANCZOS)
        self.best_compare_img = Image.open(best_image).resize((400, 400), Image.LANCZOS)
        self.update_compare_slider(0)
        img = Image.open(best_image).resize((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        self.canvas.configure(image=img_tk)
        self.canvas.image = img_tk
        self.result_label.config(text=f"Best image: {os.path.basename(best_image)}")
        self.progress.stop()
        self.progress.pack_forget()
        self.show_thumbnails()

    def show_all_images(self):
        """Show all generated images in a loop."""
        if not self.generated_images:
            return
        self.show_image_loop()

    def show_image_loop(self):
        """Loop through the generated images and display them."""
        if not self.generated_images:
            return
        img_path = self.generated_images[self.current_image_idx]
        img = Image.open(img_path).resize((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        self.canvas.configure(image=img_tk)
        self.canvas.image = img_tk
        self.result_label.config(text=f"Image {self.current_image_idx + 1} of {len(self.generated_images)}: {os.path.basename(img_path)}")
        self.current_image_idx = (self.current_image_idx + 1) % len(self.generated_images)
        self.root.after(1500, self.show_image_loop)

    def show_thumbnails(self):
        """Show generated images as thumbnails in a grid."""
        for label in self.thumb_labels:
            label.destroy()
        self.thumb_labels = []
        if not self.generated_images:
            return
        cols = 3
        for idx, img_path in enumerate(self.generated_images):
            img = Image.open(img_path).resize((100, 100))
            img_tk = ImageTk.PhotoImage(img)
            lbl = tk.Label(self.thumb_frame, image=img_tk, bg="#393E46")
            lbl.image = img_tk
            lbl.grid(row=idx // cols, column=idx % cols, padx=5, pady=5)
            self.thumb_labels.append(lbl)

    def update_compare_slider(self, val):
        if self.input_compare_img is None or self.best_compare_img is None:
            return
        alpha = float(val)
        blended = Image.blend(self.input_compare_img, self.best_compare_img, alpha)
        self.compare_img_tk = ImageTk.PhotoImage(blended)
        self.compare_canvas.delete("all")
        self.compare_canvas.create_image(200, 200, image=self.compare_img_tk)

# ---- Main ----
if __name__ == "__main__":
    root = tb.Window(themename="minty")
    app = TattooApp(root)
    root.mainloop()