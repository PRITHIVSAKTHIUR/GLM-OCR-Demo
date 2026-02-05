# **GLM-OCR-Demo**

> A Gradio-based demonstration for the zai-org/GLM-OCR multimodal OCR model. Supports text, formula, and table recognition from uploaded images, with outputs in plain text and markdown formats. Features custom HotPink theme, GPU acceleration, image orientation handling (EXIF transpose), and temporary file management for processing.

<img width="1919" height="1149" alt="Screenshot 2026-02-05 at 09-58-30 GLM OCR Demo - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/a7d968d6-95b4-495c-b323-c797077d695e" />

## Features
- **Recognition Types**: Text Recognition, Formula Recognition, Table Recognition with predefined prompts.
- **Image Handling**: Supports upload/clipboard sources; auto-converts RGBA/LA/P modes to RGB; handles EXIF orientation.
- **Outputs**: Dual tabs for plain text and markdown rendering.
- **Custom Theme**: HotPinkTheme with responsive, animated styling via CSS.
- **GPU Inference**: Uses spaces.GPU decorator for efficient processing.
- **Examples**: 5 curated images for quick testing.
- **Queueing**: Up to 50 concurrent jobs.

## Prerequisites
- Python 3.10 or higher.
- CUDA-compatible GPU (recommended for bfloat16 inference).
- Stable internet for initial model downloads from Hugging Face.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/PRITHIVSAKTHIUR/GLM-OCR-Demo.git
   cd GLM-OCR-Demo
   ```
2. Install dependencies:
   First, install pre-requirements:
   ```
   pip install -r pre-requirements.txt
   ```
   Then, install main requirements:
   ```
   pip install -r requirements.txt
   ```
   **pre-requirements.txt content:**
   ```
   pip>=23.0.0
   ```
   **requirements.txt content:**
   ```
   flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
   git+https://github.com/huggingface/transformers.git
   git+https://github.com/huggingface/accelerate.git
   git+https://github.com/huggingface/peft.git
   huggingface_hub
   sentencepiece
   opencv-python
   torch==2.6.0
   torchvision
   matplotlib
   markdown
   requests
   hf_xet
   spaces
   pillow
   gradio #@gradio6
   av
   ```
3. Start the application:
   ```
   python app.py
   ```
   The demo launches at `http://localhost:7860`.

## Usage
1. **Upload Image**: Add an image via upload or clipboard.
2. **Select Task**: Choose Text, Formula, or Table recognition.
3. **Recognize**: Click "Recognize" to process.
4. **View Outputs**: Check results in Text or Markdown tabs.

### Supported Tasks
| Task     | Prompt                  |
|----------|-------------------------|
| Text    | "Text Recognition:"    |
| Formula | "Formula Recognition:" |
| Table   | "Table Recognition:"   |

## Examples
| Input Image    | Task   |
|----------------|--------|
| examples/1.jpg | Text  |
| examples/4.jpg | Text  |
| examples/5.webp| Text  |
| examples/2.jpg | Formula |
| examples/3.jpg | Table |

## Troubleshooting
- **Model Loading**: First run downloads GLM-OCR; monitor console.
- **Image Errors**: Ensure valid RGB images; check console for processing issues.
- **OOM**: Use smaller images or reduce max_new_tokens (default 8192).
- **No Output**: Upload image first; select task.
- **Flash Attention**: Requires compatible CUDA; fallback if fails.

## Contributing
Contributions welcome! Add new tasks to `TASK_PROMPTS`, enhance CSS, or improve processing. Submit pull requests via the repository.

Repository: [https://github.com/PRITHIVSAKTHIUR/GLM-OCR-Demo.git](https://github.com/PRITHIVSAKTHIUR/GLM-OCR-Demo.git)

## License
Apache License 2.0. See [LICENSE](LICENSE) for details.
Built by Prithiv Sakthi. Report issues via the repository.
