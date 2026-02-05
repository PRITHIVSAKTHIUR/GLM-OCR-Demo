import gradio as gr
import torch
import spaces
import os
import tempfile
from PIL import Image, ImageOps
from threading import Thread
from typing import Iterable
from transformers import AutoProcessor, AutoModelForImageTextToText

from transformers.image_utils import load_image
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.hot_pink = colors.Color(
    name="hot_pink",
    c50="#FFF0F5",
    c100="#FFE4EC",
    c200="#FFC0D9",
    c300="#FF99C4",
    c400="#FF7EB8",
    c500="#FF69B4",
    c600="#E55AA0",
    c700="#CC4C8C",
    c800="#B33D78",
    c900="#992F64",
    c950="#802050",
)

class HotPinkTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.hot_pink,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

hot_pink_theme = HotPinkTheme()

MODEL_PATH = "zai-org/GLM-OCR"

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

TASK_PROMPTS = {
    "Text": "Text Recognition:",
    "Formula": "Formula Recognition:",
    "Table": "Table Recognition:",
}

@spaces.GPU
def process_image(image, task):
    if image is None:
        return "Please upload an image first"
    
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image = ImageOps.exif_transpose(image)
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    image.save(tmp.name, 'PNG')
    tmp.close()
    
    prompt = TASK_PROMPTS.get(task, "Text Recognition:")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": tmp.name},
                {"type": "text", "text": prompt}
            ],
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    
    inputs.pop("token_type_ids", None)
    
    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    output_text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )
    
    os.unlink(tmp.name)
    
    return output_text.strip()


css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* Background grid pattern - Hot Pink theme */
body, .gradio-container {
    background-color: #FFF0F5 !important;
    background-image: 
        linear-gradient(#FFC0D9 1px, transparent 1px), 
        linear-gradient(90deg, #FFC0D9 1px, transparent 1px) !important;
    background-size: 40px 40px !important;
    font-family: 'Outfit', sans-serif !important;
}

/* Dark mode grid */
.dark body, .dark .gradio-container {
    background-color: #1a1a1a !important;
    background-image: 
        linear-gradient(rgba(255, 105, 180, 0.1) 1px, transparent 1px), 
        linear-gradient(90deg, rgba(255, 105, 180, 0.1) 1px, transparent 1px) !important;
    background-size: 40px 40px !important;
}

#col-container {
    margin: 0 auto;
    max-width: 1000px;
}

/* Main title styling */
#main-title {
    text-align: center !important;
    padding: 1rem 0 0.5rem 0;
}

#main-title h1 {
    font-size: 2.5em !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #FF69B4 0%, #FF99C4 50%, #E55AA0 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradient-shift 4s ease infinite;
    letter-spacing: -0.02em;
}

@keyframes gradient-shift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

/* Subtitle styling */
#subtitle {
    text-align: center !important;
    margin-bottom: 1.5rem;
}

#subtitle p {
    margin: 0 auto;
    color: #666666;
    font-size: 1rem;
}

#subtitle a {
    color: #FF69B4 !important;
    text-decoration: none;
    font-weight: 500;
}

#subtitle a:hover {
    text-decoration: underline;
}

/* Card styling */
.gradio-group {
    background: rgba(255, 255, 255, 0.9) !important;
    border: 2px solid #FFC0D9 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 24px rgba(255, 105, 180, 0.08) !important;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.gradio-group:hover {
    box-shadow: 0 8px 32px rgba(255, 105, 180, 0.12) !important;
    border-color: #FF99C4 !important;
}

.dark .gradio-group {
    background: rgba(30, 30, 30, 0.9) !important;
    border-color: rgba(255, 105, 180, 0.3) !important;
}

/* Image upload area */
.gradio-image {
    border-radius: 10px !important;
    overflow: hidden;
    border: 2px dashed #FF99C4 !important;
    transition: all 0.3s ease;
}

.gradio-image:hover {
    border-color: #FF69B4 !important;
    background: rgba(255, 105, 180, 0.02) !important;
}

/* Radio buttons */
.gradio-radio {
    border-radius: 8px !important;
}

.gradio-radio label {
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
    border: 1px solid transparent !important;
}

.gradio-radio label:hover {
    background: rgba(255, 105, 180, 0.05) !important;
}

.gradio-radio label.selected {
    background: rgba(255, 105, 180, 0.1) !important;
    border-color: #FF69B4 !important;
}

/* Primary button */
.primary {
    border-radius: 8px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    transition: all 0.3s ease !important;
}

.primary:hover {
    transform: translateY(-2px) !important;
}

/* Tabs styling */
.tab-nav {
    border-bottom: 2px solid #FFC0D9 !important;
}

.tab-nav button {
    font-weight: 500 !important;
    padding: 10px 18px !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s ease !important;
}

.tab-nav button.selected {
    background: rgba(255, 105, 180, 0.1) !important;
    border-bottom: 2px solid #FF69B4 !important;
}

/* Output textbox */
.gradio-textbox textarea {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
    background: rgba(255, 255, 255, 0.95) !important;
    border: 1px solid #FFC0D9 !important;
    border-radius: 8px !important;
}

.dark .gradio-textbox textarea {
    background: rgba(30, 30, 30, 0.95) !important;
    border-color: rgba(255, 105, 180, 0.2) !important;
}

/* Markdown output */
.gradio-markdown {
    font-family: 'Outfit', sans-serif !important;
    line-height: 1.7 !important;
}

.gradio-markdown code {
    font-family: 'IBM Plex Mono', monospace !important;
    background: rgba(255, 105, 180, 0.08) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    color: #CC4C8C !important;
}

.gradio-markdown pre {
    background: rgba(255, 105, 180, 0.05) !important;
    border: 1px solid #FFC0D9 !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}

/* Examples section */
.gradio-examples {
    border-radius: 10px !important;
}

.gradio-examples .gallery-item {
    border: 2px solid #FFC0D9 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

.gradio-examples .gallery-item:hover {
    border-color: #FF69B4 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(255, 105, 180, 0.15) !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 105, 180, 0.05);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #FF69B4, #FF99C4);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #E55AA0, #FF69B4);
}

/* Accordion styling */
.gradio-accordion {
    border-radius: 10px !important;
    border: 1px solid #FFC0D9 !important;
}

.gradio-accordion > .label-wrap {
    background: rgba(255, 105, 180, 0.03) !important;
    border-radius: 10px !important;
}

/* Hide footer */
footer {
    display: none !important;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.gradio-row {
    animation: fadeIn 0.4s ease-out;
}

/* Label styling */
label {
    font-weight: 600 !important;
    color: #333 !important;
}

.dark label {
    color: #eee !important;
}
"""

with gr.Blocks() as demo:
    
    gr.Markdown("# **GLM-OCR**", elem_id="main-title")
    gr.Markdown("*A multimodal [OCR model](https://huggingface.co/zai-org/GLM-OCR) for complex document understanding.*", elem_id="subtitle")
    
    with gr.Row():
        
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Image",
                sources=["upload", "clipboard"],
                height=300
            )
            with gr.Row():
                task = gr.Radio(
                    choices=list(TASK_PROMPTS.keys()),
                    value="Text",
                    label="Recognition Type"
                )

            with gr.Row():
                btn = gr.Button("Recognize", variant="primary")
            
            gr.Examples(
                examples=["examples/1.jpg", "examples/4.jpg", "examples/5.webp", "examples/2.jpg", "examples/3.jpg"],
                inputs=image_input,
                label="Examples"
            )
        
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("Text"):
                    output_text = gr.Textbox(
                        label="Output",
                        lines=18,
                        interactive=True,
                    )
                
                with gr.Tab("Markdown"):
                    output_md = gr.Markdown(value="")
    
    def run_ocr(image, task):
        result = process_image(image, task)
        return result, result
    
    btn.click(
        run_ocr,
        [image_input, task],
        [output_text, output_md]
    )
    
    image_input.change(
        lambda: ("", ""),
        None,
        [output_text, output_md]
    )

if __name__ == "__main__":
    demo.queue(max_size=50).launch(css=css, theme=hot_pink_theme, mcp_server=True, ssr_mode=False, show_error=True)