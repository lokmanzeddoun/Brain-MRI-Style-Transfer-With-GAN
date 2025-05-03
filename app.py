import streamlit as st
import numpy as np
import nibabel as nib
import os
from PIL import Image
import tempfile
import shutil
from pathlib import Path
import time
import matplotlib.pyplot as plt
from utils import load_image, preprocess_image, save_image, process_image, load_model

# Page configuration
st.set_page_config(
    page_title="Brain MRI Style Transfer",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        color: #333333;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .upload-section {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .image-container {
        display: flex;
        justify-content: space-around;
        margin: 20px 0;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
    }
    .stMarkdown {
        color: #333333;
    }
    .stTextInput>div>div>input {
        color: #333333;
    }
    .stSelectbox>div>div>select {
        color: #333333;
    }
    .stFileUploader>div>div>div>div>div>div {
        color: #333333;
    }
    .stExpander>div>div>div>div>div {
        color: #333333;
    }
    .stAlert {
        background-color: #f8f9fa;
        color: #333333;
    }
    .stProgress>div>div>div>div {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Session state initialization
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'model' not in st.session_state:
    st.session_state.model = None

def display_image(image_data, title):
    """Display medical image with matplotlib."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_data, cmap='gray')
    ax.set_title(title, color='#333333', pad=10)
    ax.axis('off')
    st.pyplot(fig)
    plt.close()

def main():
    st.title("🧠 Brain MRI Style Transfer")
    st.markdown("### Transform your brain MRI scans with AI-powered style transfer")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload & Process", "Gallery", "Account"])
    
    if page == "Upload & Process":
        upload_and_process_page()
    elif page == "Gallery":
        gallery_page()
    else:
        account_page()

def upload_and_process_page():
    st.header("Upload & Process")
    
    # File upload section
    with st.container():
        st.markdown("### Upload your brain MRI scan")
        uploaded_file = st.file_uploader(
            "Choose a DICOM or NIfTI file",
            type=['dcm', 'nii', 'nii.gz'],
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                shutil.copyfileobj(uploaded_file, tmp_file)
                tmp_path = tmp_file.name
            
            try:
                # Load and display original image
                image_data = load_image(tmp_path)
                st.session_state.current_image = image_data
                
                st.markdown("### Preview")
                display_image(image_data, "Original Image")
                
                # Style selection
                st.markdown("### Select Style")
                style_options = ["T1 to T2", "T2 to T1", "Custom Style"]
                selected_style = st.selectbox("Choose the target style", style_options)
                
                # Process button
                if st.button("Process Image"):
                    with st.spinner("Processing your image..."):
                        # Load model if not already loaded
                        if st.session_state.model is None:
                            st.session_state.model = load_model("path_to_your_model.pth")
                        
                        # Process image
                        processed_image = process_image(
                            st.session_state.model,
                            image_data,
                            selected_style
                        )
                        
                        # Save processed image
                        output_path = f"processed_{Path(uploaded_file.name).stem}.png"
                        save_image(processed_image, output_path)
                        
                        # Add to processed images
                        st.session_state.processed_images.append({
                            'original': tmp_path,
                            'processed': output_path,
                            'style': selected_style
                        })
                        
                        st.success("Processing complete!")
                        
                        # Display results
                        st.markdown("### Results")
                        col1, col2 = st.columns(2)
                        with col1:
                            display_image(image_data, "Original")
                        with col2:
                            display_image(processed_image, f"Processed ({selected_style})")
                        
                        # Download button
                        with open(output_path, "rb") as file:
                            st.download_button(
                                label="Download Processed Image",
                                data=file,
                                file_name=output_path,
                                mime="image/png"
                            )
            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

def gallery_page():
    st.header("Gallery")
    st.markdown("### Your processed images")
    
    if not st.session_state.processed_images:
        st.info("No processed images available. Upload and process some images first!")
    else:
        for idx, img_data in enumerate(st.session_state.processed_images):
            with st.expander(f"Image {idx + 1} - {img_data['style']}"):
                col1, col2 = st.columns(2)
                with col1:
                    display_image(load_image(img_data['original']), "Original")
                with col2:
                    display_image(load_image(img_data['processed']), "Processed")
                
                # Download button
                with open(img_data['processed'], "rb") as file:
                    st.download_button(
                        label=f"Download Processed Image {idx + 1}",
                        data=file,
                        file_name=Path(img_data['processed']).name,
                        mime="image/png"
                    )

def account_page():
    st.header("Account Settings")
    st.markdown("### Manage your account")
    
    # TODO: Implement account management features
    st.info("Account management features coming soon!")

if __name__ == "__main__":
    main()
