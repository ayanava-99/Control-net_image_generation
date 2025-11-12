import streamlit as st
from PIL import Image
import torch
import numpy as np
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers import EulerAncestralDiscreteScheduler
import cv2
import io


st.set_page_config(
    page_title="Pro Art Anime",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Pro Art Anime")
st.markdown("Transform rough anime sketches into anime-style illustrations")

# Sidebar controls
st.sidebar.header("⚙️ Parameters")
prompt = st.sidebar.text_input(
    "Describe your anime character/scene:",
    value="1girl, solo, anime style, beautiful, detailed, high quality, masterpiece"
)

negative_prompt = st.sidebar.text_input(
    "What to avoid:",
    value="lowres, bad anatomy, bad hands, missing fingers, worst quality, low quality"
)

num_steps = st.sidebar.slider("Steps (Higher==Better quality|| Lower==faster result)", 15, 50, 30)
guidance_scale = st.sidebar.slider("Guidance Scale (Prompt strength)", 1.0, 15.0, 7.5)
controlnet_scale = st.sidebar.slider("Sketch Control Strength", 0.5, 2.0, 1.0)

# gpu
if torch.cuda.is_available():
    device ="cuda"
    dtype = torch.float16
else:
    device ="cpu"
    dtype =torch.float32
#print(device)

# Cache models 
@st.cache_resource
def load_models():
 
    try:
        st.info("⏳ Loading models... This may take a while....")
        
        #ControlNet
        controlnet = ControlNetModel.from_pretrained(
            "xinsir/anime-painter",
            torch_dtype=dtype
        )
        
        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            "gsdf/CounterfeitXL",
            subfolder="vae",
            torch_dtype=dtype
        )
        
        #scheduler
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            "gsdf/CounterfeitXL",
            subfolder="scheduler"
        )
        
        # Load pipeline
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "gsdf/CounterfeitXL",
            controlnet=controlnet,
            vae=vae,
            scheduler=scheduler,
            safety_checker=None,
            torch_dtype=dtype
        )
        
        # Optimize pipeline
        if device == "cuda":
            pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_model_cpu_offload()
        
        return pipe
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def preprocess_sketch(image_pil):
    # Convert to grayscale
    image_gray = image_pil.convert('L')
    
    # Apply threshold to create binary image
    image_array = np.array(image_gray)
    _, binary = cv2.threshold(image_array, 127, 255, cv2.THRESH_BINARY)
    
    # Apply slight blur to smooth lines
    binary = cv2.GaussianBlur(binary, (3, 3), 0)
    
    # Invert
    if np.mean(binary) > 127:
        binary = 255 - binary
    
    return Image.fromarray(binary)


def generate_anime(pipe, sketch_image, prompt, negative_prompt, num_steps, guidance_scale, controlnet_scale):

    try:
        width, height = sketch_image.size
        #dimensions multiples of 8
        width =(width // 8) * 8
        height= (height //8)* 8
        sketch_image = sketch_image.resize((width, height))
        
        # Generate with ControlNet
        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                image=sketch_image,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_scale,
                height=height,
                width=width
            )
        
        return output.images[0]
    
    except Exception as e:
        st.error(f"Error during generation: {str(e)}")
        return None

# Main UI
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Your Sketch")
    uploaded_file = st.file_uploader(
        "Choose a sketch image (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file is not None:
        sketch = Image.open(uploaded_file)
        st.image(sketch, caption="Original Sketch", use_column_width=True)

with col2:
    st.subheader("🎬 Generated Anime")
    if uploaded_file is not None:
        if st.button("✨ Generate Anime Illustration", key="generate"):
            # Load models
            pipe = load_models()
            
            if pipe is not None:
                with st.spinner("🔄 Processing your sketch..."):
                    # Preprocess sketch
                    sketch_pil = Image.open(uploaded_file)
                    processed_sketch = preprocess_sketch(sketch_pil)
                    
                    # Show processed sketch
                    st.caption("Processed Sketch")
                    st.image(processed_sketch, use_column_width=True)
                    
                    # Generate anime
                    result = generate_anime(
                        pipe,
                        processed_sketch,
                        prompt,
                        negative_prompt,
                        num_steps,
                        guidance_scale,
                        controlnet_scale
                    )
                    
                    if result is not None:
                        st.success("✅ Generation complete!")
                        st.image(result, caption="Generated Anime", use_column_width=True)
                        
                        # Download button
                        buf = io.BytesIO()
                        result.save(buf, format="PNG")
                        buf.seek(0)
                        st.download_button(
                            label="⬇️ Download Image",
                            data=buf,
                            file_name="anime_output.png",
                            mime="image/png"
                        )
    else:
        st.info("Upload a sketch to get started!")

# Footer
st.divider()
st.markdown("""
### 📝 How to use:
1. **Upload** your sketch (hand-drawn or digital)
2. **Write** a prompt describing your desired anime character/scene
3. **Adjust** settings to fine-tune the output
4. **Click** "Generate Anime Illustration" button
5. **Download** your beautiful anime art!

### 💡 Tips:
- Clear, high-contrast sketches work best
- Be specific in your prompts for better results
- Higher inference steps = better quality but slower
- Adjust guidance scale for prompt 

### 🔧 Technical Details:
- Model: xinsir/anime-painter (ControlNet)
- Base: Stable Diffusion XL (CounterfeitXL)
- Framework: Diffusers library
""")
