import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Set page config
st.set_page_config(page_title="Brain MRI Style Transfer", layout="centered")

st.title("Brain MRI Style Transfer with CycleGAN")
st.write("Upload a T1 or T2 MRI image (PNG/JPG), select the transfer direction, and see the result.")

# File uploader
uploaded_file = st.file_uploader("Choose a PNG or JPG MRI image", type=["png", "jpg", "jpeg"])

# Direction selector
direction = st.radio(
    "Select style transfer direction:",
    ("T1 → T2", "T2 → T1")
)

# Model file mapping
generator_files = {
    "T1 → T2": "generator_t1_to_t2.keras",
    "T2 → T1": "generator_t2_to_t1.keras"
}

@tf.keras.utils.register_keras_serializable()
class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True
        )
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True
        )

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

def preprocess_image(image: Image.Image) -> np.ndarray:
    # Convert to grayscale
    image = image.convert("L")
    # Resize to 64x64
    image = image.resize((64, 64))
    # Convert to numpy array
    img_array = np.array(image).astype(np.float32)
    # Normalize to [-1, 1]
    img_array = (img_array / 127.5) - 1.0
    # Add channel dimension
    img_array = img_array.reshape((64, 64, 1))
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def denormalize_image(img: np.ndarray) -> np.ndarray:
    # Remove batch and channel dimensions if present
    if img.ndim == 4:
        img = img[0]
    if img.shape[-1] == 1:
        img = img[..., 0]
    # Denormalize from [-1, 1] to [0, 255]
    img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return img

if uploaded_file is not None:
    # Load image
    input_image = Image.open(uploaded_file)
    st.write("### Input vs. Style-Transferred Output")

    # Preprocess
    preprocessed = preprocess_image(input_image)

    # Load model
    model_path = generator_files[direction]
    with st.spinner(f"Loading model: {model_path}..."):
        generator = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={'InstanceNormalization': InstanceNormalization}
        )

    # Run inference
    with st.spinner("Generating style-transferred image..."):
        output = generator.predict(preprocessed)
        output_img = denormalize_image(output)

    # Convert output to PIL Image and upscale to input size for clarity
    output_pil = Image.fromarray(output_img)
    output_pil = output_pil.resize(input_image.size, resample=Image.BICUBIC)

    # Show input and output side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(input_image, caption="Input Image", use_column_width=True)
    with col2:
        st.image(output_pil, caption=f"{direction} Result", use_column_width=True)
