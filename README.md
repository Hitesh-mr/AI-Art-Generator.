!pip install gradio diffusers transformers accelerate --quiet

import gradio as gr
from diffusers import StableDiffusionPipeline
import torch
import random

# Load model
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    revision="fp16"
)
pipe = pipe.to("cuda")

# Style list
STYLES = {
    "None": "",
    "Anime": "anime style, colorful, dramatic lighting",
    "Cyberpunk": "cyberpunk, neon lights, futuristic city, high contrast",
    "Watercolor": "watercolor painting, soft brush strokes, pastel colors",
    "Dark Fantasy": "dark fantasy, gothic, cinematic lighting, highly detailed",
    "Synthwave": "80s synthwave, neon grid, retro-futuristic, glowing",
}

# Random prompts
RANDOM_PROMPTS = [
    "A futuristic samurai in a neon-lit city",
    "A dragon made of clouds and stars",
    "A robot painting the moon on a canvas",
    "A forest made of candy and chocolate",
    "A floating island with ancient ruins",
    "A castle built into a giant tree",
    "A dreamscape with impossible geometry",
]

# AI generate function
def generate(prompt, style):
    if style != "None":
        prompt = f"{prompt}, {STYLES[style]}"
    image = pipe(prompt).images[0]
    return image

def surprise_me():
    return random.choice(RANDOM_PROMPTS)

# 🔥 Neon animated CSS
custom_css = """
body {
    background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #0f0c29);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
h1, label {
    color: #ffffff !important;
    text-shadow: 0 0 10px #ff00cc;
}
.gr-button {
    background: transparent;
    color: white;
    border: 2px solid #ff00cc;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
    transition: 0.3s ease;
    box-shadow: 0 0 10px #ff00cc;
}
.gr-button:hover {
    background: #ff00cc;
    color: black;
    box-shadow: 0 0 20px #ff00cc, 0 0 40px #ff00cc;
}
"""

# Gradio App UI
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1 style='text-align: center;'>⚡ Rishi's AI Art Generator</h1>")

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="🖋️ Prompt", placeholder="e.g. A neon samurai cat in Tokyo")
            style_dropdown = gr.Dropdown(choices=list(STYLES.keys()), value="None", label="🎨 Art Style")
            with gr.Row():
                generate_btn = gr.Button("✨ Generate")
                surprise_btn = gr.Button("🎲 Surprise Me")

        with gr.Column():
            output_image = gr.Image(label="🎨 AI Art", type="pil", show_label=True)

    generate_btn.click(fn=generate, inputs=[prompt_input, style_dropdown], outputs=output_image)
    surprise_btn.click(fn=surprise_me, outputs=prompt_input)

demo.launch()
