# Cell 1: Imports and Setup
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from PIL import Image
import os
print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
#Define the custom DepthToSpace layer for ESPCN
class DepthToSpace(layers.Layer):
    def __init__(self, scale, **kwargs):
        super(DepthToSpace, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.scale)
#Define ESPCN model
def espcn_model(scale):
    inputs = keras.Input(shape=(None, None, 1))
    x = layers.Conv2D(64, 5, padding="same", activation="tanh")(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="tanh")(x)
    x = layers.Conv2D(scale**2, 3, padding="same", activation="tanh")(x)
    outputs = DepthToSpace(scale)(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model
#Define SRCNN model
def srcnn_model():
    SRCNN = Sequential()
    SRCNN.add(Conv2D(128, (9, 9), activation='relu', padding='valid', input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    SRCNN.add(Conv2D(1, (5, 5), activation='linear', padding='valid'))
    adam = Adam(learning_rate=0.0003)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN
#Load Pre-trained Weights
def load_pretrained_weights(model, model_type):
    try:
        if model_type == 'ESPCN':
            model.load_weights('3051crop_weight_200.h5')  # Ensure this path is correct
        elif model_type == 'SRCNN':
            model.load_weights('3051crop_weight_200.h5')  # Ensure this path is correct
        print(f"{model_type} model loaded successfully with pre-trained weights.")
    except Exception as e:
        print(f"Error loading weights for {model_type}: {str(e)}")
    return model
#Image Processing and Degradation Functions
import cv2
import numpy as np
from PIL import Image

def preprocess_image(uploaded_image, scale):
    # Convert uploaded image to NumPy array
    img = cv2.cvtColor(uploaded_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV processing

    # Convert the image to YCrCb color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    # Split the channels
    y, cr, cb = cv2.split(img)
    
    # Normalize the Y channel
    y = y.astype(np.float32) / 255.0
    
    y = modcrop(y, scale)
    cr = modcrop(cr, scale)
    cb = modcrop(cb, scale)
    
    return y, cr, cb
def modcrop(img, scale):
    h, w = img.shape[:2]
    h = h - h % scale
    w = w - w % scale
    return img[:h, :w]
def shave(image, border):
    return image[border:-border, border:-border]
def degrade_image(y, factor):
    h, w = y.shape
    degraded_y = cv2.resize(y, (w // factor, h // factor), interpolation=cv2.INTER_LINEAR)
    degraded_y = cv2.resize(degraded_y, (w, h), interpolation=cv2.INTER_LINEAR)
    return degraded_y

def upscale_image(model, lr_image, scale):
    # Normalize low-resolution image to [0, 1]
    #lr_image = lr_image / 255.0
    lr_image = lr_image[np.newaxis, ..., np.newaxis]
    sr_image = model.predict(lr_image)[0]
    sr_image = np.clip(sr_image, 0, 1) * 255
    sr_image = sr_image.astype(np.uint8)
    return sr_image
#Evaluation Metrics (PSNR, MSE, SSIM)
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2):
    mse_value = mse(img1, img2)
    if mse_value == 0:
        return 100  # Images are identical
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse_value))

def calculate_ssim(img1, img2):
    # Set the maximum window size based on the smallest image dimension
    min_dim = min(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1])
    
    # Ensure win_size is odd and less than or equal to min_dim
    win_size = min(min_dim, 7)  # Default win_size is 7; adjust if image is smaller
    if win_size % 2 == 0:  # Ensure win_size is odd
        win_size -= 1
    
    # Handle multichannel images by setting channel_axis
    return ssim(img1, img2, multichannel=True, win_size=win_size, channel_axis=2)
def compare_images(target, ref):
    psnr_value = calculate_psnr(target, ref)
    ssim_value = calculate_ssim(target, ref)
    mse_value = mse(target, ref)
    return psnr_value, mse_value, ssim_value
def mse(target, ref):
    err = np.sum((target.astype('float') - ref.astype('float')) ** 2)
    err /= float(target.shape[0] * target.shape[1])
    return err
def enhance_image(image_path, model_type, scale, degradation_factor):
    # Preprocess the image to get Y, Cr, and Cb channels
    y, cr, cb = preprocess_image(np.array(image_path), scale)
    degraded_y = degrade_image(y, degradation_factor)

    # Load the model (ESPCN or SRCNN)
    if model_type == 'ESPCN':
        model = espcn_model(scale)
    else:
        model = srcnn_model()

    # Load pre-trained weights
    model = load_pretrained_weights(model, model_type)

    # Perform super-resolution on degraded image (Y channel)
    sr_y = upscale_image(model, degraded_y, scale)

    # Resize the `sr_y` to match the dimensions of `cr` and `cb`
    h, w = cr.shape
    sr_y = cv2.resize(sr_y, (w, h), interpolation=cv2.INTER_LINEAR)

    # Merge the Y, Cr, and Cb channels back into an image
    sr_image = cv2.merge([sr_y, cr, cb])
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_YCrCb2RGB)

    # Load the original image and convert it to RGB
    # original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_image = np.array(image_path)

    # Ensure that the dimensions of both images are the same
    if sr_image.shape != original_image.shape:
        # Resize the original image to match the super-resolved image dimensions
        original_image = cv2.resize(original_image, (sr_image.shape[1], sr_image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Calculate PSNR, SSIM, and MSE
    psnr_value = calculate_psnr(original_image, sr_image)
    ssim_value = calculate_ssim(original_image, sr_image)  # Pass the modified calculate_ssim
    mse_value = mse(original_image, sr_image)

    return original_image, degraded_y, sr_image, psnr_value, ssim_value, mse_value
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.models import load_model


def main():
    st.title("Super-Resolution Image Enhancement")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])
    
    model_choice = st.selectbox("Choose Model", ["SRCNN", "ESPCN"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        scale = st.slider("Upscaling Factor", 2, 4, 3)
        degradation_factor = st.slider("Degradation Factor", 2, 10, 3)

        if st.button("Enhance Image"):
            st.write("Processing...")
            img_array = np.array(image)

            # Assuming you have an `enhance_image` function defined
            original, degraded, enhanced, psnr_value, ssim_value, mse_value = enhance_image(img_array, model_choice, scale, degradation_factor)
            
            st.image([original, degraded, enhanced], caption=["Original", "Degraded", "Enhanced"], use_column_width=True)
            st.write(f"PSNR: {psnr_value}")
            st.write(f"SSIM: {ssim_value}")
            st.write(f"MSE: {mse_value}")

if __name__ == "__main__":
    main()
