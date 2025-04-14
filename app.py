import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os
from utils.processing import pop_art_style, inksplash_style, sketch_style, ascii_art
from utils.compression import compress_image_dct

# Set page configuration with custom theme
st.set_page_config(
    page_title="Artistic Image Transformer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to style the app
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: black;
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3rem;
        font-weight: bold;
        background-color: black;
        color: white;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #5549c0;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    .style-option:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 {
        color: #2d3436;
        font-weight: 800;
    }
    .sidebar .stRadio > div {
        padding: 10px;
        background-color: white;
        border-radius: 12px;
        margin-bottom: 10px;
    }
    .image-container {
        background-color: white;
        border-radius: 12px;
        padding: 15px;
        color: black;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .download-btn {
        display: inline-block;
        background-color: #00b894;
        color: black;
        padding: 10px 15px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: bold;
        margin-top: 10px;
        transition: all 0.3s ease;
    }

    .download-btn:hover {
        background-color: #00a08a;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Function to generate download link for images
def get_image_download_link(img, filename, link_text):
    """Generate a download link for an image"""
    buffered = io.BytesIO()
    
    # Convert from OpenCV BGR to RGB if needed
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
    
    # Convert to PIL Image
    pil_img = Image.fromarray(img_rgb)
    pil_img.save(buffered, format="PNG")
    
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}" class="download-btn">{link_text}</a>'
    return href

# Title
st.title("🎨 Artistic Image Transformer")
st.markdown("Transform your photos into stunning artistic styles")

# Sidebar for controls
with st.sidebar:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        st.success("✅ Image uploaded successfully!")
        
        # Style selection
        st.header("🖌️ Select Style")
        style_option = st.radio(
            "Choose your artistic style:",
            ["Pop Art", "Ink Splash", "Sketch", "ASCII Art", "Compressed"]
        )
        
        # Style-specific settings
        if style_option == "Pop Art":
            st.subheader("Pop Art Settings")
            colors = st.slider("Number of Colors", 3, 8, 5)
            background = st.selectbox(
                "Background Color", 
                ["Orange", "Red", "Blue", "Yellow", "Green", "Purple"]
            )
            
        elif style_option == "Sketch":
            st.subheader("Sketch Settings")
            intensity = st.slider("Pencil Intensity", 0.2, 2.0, 0.8)
            
        elif style_option == "ASCII Art":
            st.subheader("ASCII Art Settings")
            width = st.slider("Width (characters)", 50, 150, 100)
            height = st.slider("Height (lines)", 20, 100, 50)
            
        elif style_option == "Compressed":
            st.subheader("Compression Settings")
            quality = st.slider("Quality", 5, 95, 20)
        
        # Generate button
        generate_btn = st.button("🔄 Generate Art")

# Main content area
if uploaded_file:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    
    # Display original image in first column
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.subheader("Original Image")
        st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Process and display transformed image when generate button is clicked
    if generate_btn:
        with col2:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            
            if style_option == "Pop Art":
                st.subheader("🎭 Pop Art Style")
                with st.spinner("Generating Pop Art..."):
                    result = pop_art_style(original_img, colors, background.lower())
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)
                
                # Download button
                st.markdown(
                    get_image_download_link(result, "pop_art.png", "💾 Download Pop Art Image"),
                    unsafe_allow_html=True
                )
                
            elif style_option == "Ink Splash":
                st.subheader("🖋️ Ink Splash Style")
                with st.spinner("Creating Ink Splash Effect..."):
                    result = inksplash_style(original_img)
                st.image(result, use_column_width=True)
                
                # Download button
                st.markdown(
                    get_image_download_link(result, "ink_splash.png", "💾 Download Ink Splash Image"),
                    unsafe_allow_html=True
                )
                
            elif style_option == "Sketch":
                st.subheader("✏️ Sketch Style")
                with st.spinner("Creating Sketch..."):
                    result = sketch_style(original_img, intensity)
                st.image(result, use_column_width=True)
                
                # Download button
                st.markdown(
                    get_image_download_link(result, "sketch.png", "💾 Download Sketch Image"),
                    unsafe_allow_html=True
                )
                
            elif style_option == "ASCII Art":
                st.subheader("📝 ASCII Art")
                with st.spinner("Generating ASCII Art..."):
                    ascii_result = ascii_art(original_img, width, height)
                st.text_area("ASCII Result", ascii_result, height=400)
                
                # Download ASCII art as text file
                b64 = base64.b64encode(ascii_result.encode()).decode()
                href = f'<a href="data:file/txt;base64,{b64}" download="ascii_art.txt" class="download-btn">💾 Download ASCII Text</a>'
                st.markdown(href, unsafe_allow_html=True)
                
            elif style_option == "Compressed":
                st.subheader("🗜️ Compressed Image")
                with st.spinner("Compressing Image..."):
                    result = compress_image_dct(original_img, quality)
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)
                
                # Download button
                st.markdown(
                    get_image_download_link(result, "compressed.png", "💾 Download Compressed Image"),
                    unsafe_allow_html=True
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
else:
    
    # Show example information in an attractive format
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="style-option">', unsafe_allow_html=True)
        st.subheader("✨ Available Styles")
        st.markdown("""
        - **Pop Art**: Vibrant colors with halftone dots, inspired by classic pop art
        - **Ink Splash**: Artistic ink splash effect with optional color
        - **Sketch**: Pencil drawing effect with adjustable intensity
        - **ASCII Art**: Convert your image to text characters
        - **Compressed**: Apply compression while maintaining quality
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="style-option">', unsafe_allow_html=True)
        st.subheader("🚀 How It Works")
        st.markdown("""
        1. Upload your image using the sidebar
        2. Select your desired artistic style
        3. Adjust the style-specific settings
        4. Click "Generate Art" to transform your image
        5. Download your masterpiece with one click
        """)
        st.markdown('</div>', unsafe_allow_html=True)
