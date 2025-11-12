# Pro Art Anime - Complete Setup Guide

## Project Overview

Transform rough hand-drawn sketches into high-quality anime-style illustrations. This Streamlit application uses **ControlNet** (xinsir/anime-painter) with **Stable Diffusion XL** to convert the sketches into anime art.

**Key Features:**
- 🎨 Upload hand-drawn sketches (PNG/JPG)
- ✨ Generate anime-style illustrations with custom prompts
- ⚙️ Adjust quality, guidance, and sketch control parameters
- ⬇️ Download generated images
- 🚀 GPU acceleration for fast generation

---

## Quick Start

### 1. Clone the Project
```bash
git clone https://github.com/ayanava-99/Control-net_image_generation.git
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## System Requirements

### Minimum
- **Python**: 3.8+
- **RAM**: 16GB
- **Disk**: 12GB (for models)
- **Processor**: Intel i7 / AMD Ryzen 7

### Recommended (for fast generation)
- **GPU**: NVIDIA RTX 3060 (12GB VRAM) or better
- **RAM**: 32GB
- **Disk**: 20GB SSD
- **CUDA**: 11.8

### Generate Times Estimate
- **CPU**: 4 - 5 minutes (30 steps)
- **GPU (RTX 3060)**: 15-30 seconds (30 steps)
- **GPU (RTX 4090)**: 5-10 seconds (30 steps)

---

## Installation Details

### GPU Setup (NVIDIA CUDA)

**1. Install CUDA Toolkit 11.8+**
https://developer.nvidia.com/cuda-downloads

**2. Install cuDNN**
https://developer.nvidia.com/cudnn

**3. Install PyTorch with CUDA Support**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**4. Verify GPU**
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### CPU-Only Setup
If you don't have a compatible GPU, the app will work on CPU (slower):
```bash
pip install -r requirements.txt
# App is optimised to automatically use the CPU
```

---

## Usage Guide

### Step 1: Upload Sketch
1. Click "Choose a sketch image" in the left panel
2. Select a PNG or JPG image of your hand-drawn sketch
3. The original sketch appears in the preview

### Step 2: Write Prompt
In the sidebar, describe your desired output:
```
Example: "1girl, blonde hair, blue eyes, happy smile, anime style, masterpiece, best quality"
```

**Good Prompt Structure:**
```
[Character]: 1girl, blonde hair, blue eyes
[Expression]: happy smile, looking at viewer
[Style]: anime style, detailed, best quality
[Setting]: standing in school, sunny day
```

### Step 3: Adjust Settings
- **Inference Steps**: 15-50 (higher = better quality, slower)
- **Guidance Scale**: 5-10 (higher = follows prompt more strictly)
- **Sketch Control**: 0.5-2.0 (higher = more sketch influence)

### Step 4: Generate
Click "✨ Generate Anime Illustration" button

### Step 5: Download
Once generated, click "⬇️ Download Image" to save as PNG

---

## Example Prompts

### Female Characters
```
"1girl, solo, blonde hair, blue eyes, happy expression, school uniform, 
 detailed face, beautiful, masterpiece, anime style, best quality"

"1girl, pink hair, shy smile, looking away, casual clothes, detailed, 
 high quality, anime, soft lighting"
```

### Male Characters
```
"1boy, solo, black hair, serious expression, wearing suit, professional, 
 detailed, best quality, anime style"

"1boy, orange hair, determined expression, warrior outfit, action pose, 
 epic, masterpiece, anime"
```

### Multiple Characters
```
"2girls, friends, smiling, embracing, beautiful faces, detailed, 
 anime style, soft colors, masterpiece"
```

### Scenes
```
"1girl standing in garden, flowers, sunny day, butterflies, 
 peaceful, beautiful, detailed, anime style"

"fantasy landscape, mountains, castle, sunset, epic, masterpiece, 
 anime style, beautiful lighting"
```

---

## Troubleshooting

### "CUDA out of memory"
**Solution**: Reduce inference steps (15-20) or use CPU
```python
# In app.py, change this line:
num_steps = st.sidebar.slider("...", 15, 25, 20)  # Max 25
```

### Models Not Downloading
**Solution**: Download manually
```bash
python -c "from diffusers import ControlNetModel; ControlNetModel.from_pretrained('xinsir/anime-painter')"
python -c "from diffusers import StableDiffusionXLControlNetPipeline; StableDiffusionXLControlNetPipeline.from_pretrained('gsdf/CounterfeitXL')"
```

### Slow Generation (Even with GPU)
**Solution**: 
- Reduce inference steps to 20
- Use smaller image dimensions (512x512)
- Close other GPU applications

### Low Quality Output
**Solution**:
- Improve your prompt (be more descriptive)
- Increase inference steps to 40-50
- Use high-contrast, clear sketches
- Increase guidance scale to 8-10

### ModuleNotFoundError
**Solution**: Reinstall dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Streamlit Not Starting
**Solution**: Check Python version and reinstall
```bash
python --version  # Should be 3.8+
pip install --upgrade streamlit
streamlit run app.py
```

---

## Model Information

### ControlNet: xinsir/anime-painter
- **Purpose**: Sketch-to-image generation conditioned on lineart
- **Training**: 1024x1024 anime illustrations
- **Type**: SDXL-based ControlNet
- **Download Size**: ~2.5GB
- **Strengths**: Anime-specific, supports variable line thickness

### Base Model: CounterfeitXL (Stable Diffusion XL)
- **Purpose**: High-quality anime image generation
- **Type**: SDXL 1.0 fine-tuned for anime
- **Download Size**: ~6.5GB
- **Quality**: 1024x1024 default resolution


---

## Advanced Configuration

### Enable Memory Efficient Settings
In `app.py`, uncomment for extremely limited VRAM:
```python
pipe.enable_sequential_cpu_offload()  
pipe.enable_attention_slicing()       
```

---

## Performance Optimization

### Fast Mode (15-20 seconds)
```python
num_steps = 20
guidance_scale = 5.0
```

### Balanced Mode (30-45 seconds)
```python
num_steps = 30
guidance_scale = 7.5
```

### Quality Mode (60-90 seconds)
```python
num_steps = 40
guidance_scale = 10.0
```

---

## File Structure
```
anime-sketch-enhancer/
├── app.py                    # Main application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── Report_APR_Group_19.pdf       # Report            
```

---

## License & Credits

- **ControlNet Model**: xinsir/anime-painter
- **Base Model**: Stable Diffusion XL (CounterfeitXL)
- **Framework**: Streamlit + Hugging Face Diffusers
- **License**: MIT (See LICENSE file)

---

Made with ❤️ using Streamlit & Stable Diffusion
