import streamlit as st
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.layers import BatchNormalization, Dropout
import shutil
from PIL import Image, ImageDraw
import tempfile
import cv2
import requests
from tqdm import tqdm
from ultralytics import YOLO

# Set page config
st.set_page_config(
    page_title="Face Finder",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        background-color: #f8f9fa;
    }
    .similarity-badge {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Download YOLO model function
def download_yolo_model(model_url, model_path):
    """Download YOLO model if it doesn't exist"""
    if not os.path.exists(model_path):
        try:
            st.info(f"Downloading YOLO model from {model_url}...")
            response = requests.get(model_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_path, 'wb') as file, tqdm(
                desc=model_path,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
            st.success("YOLO model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading YOLO model: {e}")
            return False
    return True

# Load YOLO face detection model
@st.cache_resource
def load_yolo_model():
    """Load YOLO face detection model"""
    try:
        # First try to import ultralytics
        try:
            from ultralytics import YOLO
        except ImportError:
            st.error("Ultralytics package not installed. Please run: pip install ultralytics")
            return None
        
        # Model URL and path
        model_url = "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt"
        model_path = "yolov8n-face.pt"
        
        # Download model if needed
        if not download_yolo_model(model_url, model_path):
            return None
        
        # Load the model
        model = YOLO(model_path)
        st.success("YOLO model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

def detect_faces_yolo(_model, img_path, confidence_threshold=0.5):
    """Detect faces in an image using YOLO"""
    try:
        # Run inference
        results = _model(img_path, conf=confidence_threshold, verbose=False)
        
        # Extract bounding boxes
        boxes = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
        
        return boxes
    except Exception as e:
        st.error(f"Error in face detection: {e}")
        return []

def extract_face(img_path, yolo_model, confidence_threshold=0.5):
    """Extract face from image and return the whole image with face bounding boxes"""
    try:
        # Load the image
        img = Image.open(img_path)
        img_array = np.array(img)
        
        # Detect faces
        boxes = detect_faces_yolo(yolo_model, img_path, confidence_threshold)
        
        if not boxes:
            return None
        
        # Draw bounding boxes on the image
        img_with_boxes = img.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        
        for box in boxes:
            x1, y1, x2, y2, conf = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1-20), f"{conf:.2f}", fill="red")
        
        return img_with_boxes
    except Exception as e:
        st.error(f"Error extracting face: {e}")
        return None

def extract_face_crop(img_path, yolo_model, confidence_threshold=0.5):
    """Extract cropped face from image (for processing)"""
    try:
        # Load the image
        img = Image.open(img_path)
        img_array = np.array(img)
        
        # Detect faces
        boxes = detect_faces_yolo(yolo_model, img_path, confidence_threshold)
        
        if not boxes:
            return None
        
        # Use the first detected face
        x1, y1, x2, y2, conf = boxes[0]
        
        # Crop the face
        face_img = img.crop((x1, y1, x2, y2))
        
        return face_img
    except Exception as e:
        st.error(f"Error extracting face crop: {e}")
        return None
def embedding_model(input_shape=(32, 32, 3), embedding_dim=128):
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = Dropout(0.2)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = Dropout(0.3)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = Dropout(0.4)(x)

    # Dense embedding
    x = layers.Flatten()(x)
    x = layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001)
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = layers.Dense(embedding_dim)(x)

    # L2-normalized output with explicit output shape
    def l2_normalize(x):
        return tf.math.l2_normalize(x, axis=1)

    outputs = layers.Lambda(l2_normalize, output_shape=(embedding_dim,), name='l2_normalization')(x)

    return Model(inputs, outputs)

# Load model function with caching
@st.cache_resource
def load_face_model(model_path):
    """Load the face recognition model"""
    try:
        # Try to load the complete model first
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'l2_normalization': lambda x: tf.math.l2_normalize(x, axis=1)}
        )
        st.success("Model loaded successfully!")
        return model
    except:
        try:
            # If that fails, create model architecture and load weights
            model = embedding_model()
            model.load_weights(model_path)
            st.success("Model weights loaded successfully!")
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

# Image processing functions
def process_image(img, target_size=(32, 32)):
    """Process image for model input"""
    try:
        if isinstance(img, str):
            # If it's a file path
            img = image.load_img(img, target_size=target_size)
        else:
            # If it's already a PIL image
            img = img.resize(target_size)
        
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        img_array = (img_array * 2) - 1  # Convert to [-1, 1]
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def extract_embedding(model, img_array):
    """Extract embedding from image array"""
    if img_array is None:
        return None
    img_array = np.expand_dims(img_array, axis=0)
    embedding = model.predict(img_array, verbose=0)[0]
    return embedding

def process_folder(model, yolo_model, folder_path, confidence_threshold=0.5):
    """Process all images in a folder, extract faces, and get embeddings"""
    image_paths = []
    embeddings = []
    face_images = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.webp', '.jpeg', '.bmp')):
            img_path = os.path.join(folder_path, filename)
            try:
                # Extract face crop from image for processing
                face_img = extract_face_crop(img_path, yolo_model, confidence_threshold)
                
                if face_img is not None:
                    img_array = process_image(face_img)
                    
                    if img_array is not None:
                        embedding = extract_embedding(model, img_array)
                        if embedding is not None:
                            image_paths.append(img_path)
                            embeddings.append(embedding)
                            # Store the full image with face detection for display
                            face_detected_img = extract_face(img_path, yolo_model, confidence_threshold)
                            face_images.append(face_detected_img if face_detected_img else Image.open(img_path))
            except Exception as e:
                st.warning(f"Could not process {filename}: {e}")
    
    return image_paths, np.array(embeddings), face_images

def find_similar_faces(model, yolo_model, query_img, database_folder, threshold=0.7, max_results=20, confidence_threshold=0.5):
    """Find all images of the same person in a database folder"""
    # Extract face from query image
    if isinstance(query_img, str):
        # If it's a file path
        query_face = extract_face_crop(query_img, yolo_model, confidence_threshold)
    else:
        # If it's already a PIL image, save temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            query_img.save(tmp.name)
            tmp_path = tmp.name
        
        query_face = extract_face_crop(tmp_path, yolo_model, confidence_threshold)
        os.unlink(tmp_path)
    
    if query_face is None:
        st.error("No face detected in query image")
        return [], None
    
    # Process query face
    query_array = process_image(query_face)
    if query_array is None:
        return [], None

    query_embedding = extract_embedding(model, query_array)
    
    # Process all images in database
    database_paths, database_embeddings, face_images = process_folder(model, yolo_model, database_folder, confidence_threshold)
    
    if len(database_embeddings) == 0:
        return [], None
    
    # Calculate distances
    distances = np.linalg.norm(database_embeddings - query_embedding, axis=1)
    
    # Find similar images
    similar_indices = np.where(distances < threshold)[0]
    
    # Sort by distance
    sorted_indices = similar_indices[np.argsort(distances[similar_indices])]
    
    # Return results with both the original image path and the processed face image
    results = [(database_paths[i], face_images[i], distances[i]) for i in sorted_indices[:max_results]]
    
    return results, query_face

# Main app
def main():
    st.markdown('<h1 class="main-header">🔍 Face Finder App</h1>', unsafe_allow_html=True)
    st.markdown("Upload a photo of a person and find all their photos in a folder")

    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'query_image' not in st.session_state:
        st.session_state.query_image = None
    if 'query_face' not in st.session_state:
        st.session_state.query_face = None

    # Sidebar for inputs
    with st.sidebar:
        st.header("Settings")
        
        # Model path
        model_path = st.text_input(
            "Model Path", 
            value="lfw_embedding_model.h5",
            help="Path to your trained model file"
        )
        
        # Database folder
        database_folder = st.text_input(
            "Database Folder", 
            value="Images_folder",
            help="Folder containing images to search through"
        )
        
        # Similarity threshold
        threshold = st.slider(
            "Similarity Threshold", 
            min_value=0.5, 
            max_value=1.0, 
            value=0.55, 
            step=0.01,
            help="Lower values = more strict matching"
        )
        
        # Face detection confidence
        confidence_threshold = st.slider(
            "Face Detection Confidence", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.5, 
            step=0.05,
            help="Confidence threshold for face detection"
        )
        
        # Max results
        max_results = st.slider(
            "Maximum Results", 
            min_value=5, 
            max_value=50, 
            value=20,
            help="Maximum number of results to show"
        )

    # Load YOLO model
    yolo_model = load_yolo_model()

    # Main content area
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Upload Query Photo")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a photo", 
            type=['jpg', 'jpeg', 'png','webp'],
            help="Upload a clear photo of the person you want to find"
        )
        
        if uploaded_file is not None and yolo_model is not None:
            # Display the uploaded image
            query_image = Image.open(uploaded_file)
            st.session_state.query_image = query_image
            
            # Extract face and show the image with detection
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                query_image.save(tmp.name)
                tmp_path = tmp.name

            # Now the file is closed and can be processed
            query_face_detected = extract_face(tmp_path, yolo_model, confidence_threshold)
            query_face_crop = extract_face_crop(tmp_path, yolo_model, confidence_threshold)
            os.unlink(tmp_path)  # Now it's safe to delete
            
            if query_face_detected:
                st.session_state.query_face = query_face_crop
                st.image(query_face_detected, caption="Detected Face", use_container_width=True)
            else:
                st.image(query_image, caption="Original Image (No Face Detected)", use_container_width=True)
            
            # Process button
            if st.button("Find Similar Faces", type="primary"):
                if not os.path.exists(database_folder):
                    st.error(f"Database folder not found: {database_folder}")
                else:
                    with st.spinner("Loading model and processing images..."):
                        # Load model
                        model = load_face_model(model_path)
                        
                        if model and yolo_model:
                            # Find similar faces
                            results, query_face = find_similar_faces(
                                model, yolo_model, query_image, database_folder, 
                                threshold, max_results, confidence_threshold
                            )
                            st.session_state.results = results
                            st.session_state.query_face = query_face

    with col2:
        st.header("Results")
        
        if st.session_state.results is not None:
            if len(st.session_state.results) > 0:
                st.success(f"Found {len(st.session_state.results)} matching images!")
                
                # Display results in a grid
                cols = st.columns(3)
                for i, (img_path, face_img, distance) in enumerate(st.session_state.results):
                    with cols[i % 3]:
                        try:
                            # Display the image with face detection
                            st.image(face_img, use_container_width=True)
                                
                            similarity = 1 - distance  # Convert distance to similarity score
                            st.markdown(f"""
                            <div class="result-card">
                                <strong>{os.path.basename(img_path)}</strong><br>
                                Similarity: <span class="similarity-badge">{similarity:.2%}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error displaying {img_path}: {e}")
                
                # Download button for results
                if st.button("Save Results to Folder"):
                    output_folder = "matched_faces"
                    os.makedirs(output_folder, exist_ok=True)
                    
                    # Copy query image
                    if st.session_state.query_face:
                        query_path = os.path.join(output_folder, "query_face.jpg")
                        st.session_state.query_face.save(query_path)
                    
                    # Copy matched images (with face detection)
                    for img_path, face_img, distance in st.session_state.results:
                        filename = os.path.basename(img_path)
                        output_path = os.path.join(output_folder, f"{distance:.3f}_{filename}")
                        face_img.save(output_path)
                    
                    st.success(f"Saved {len(st.session_state.results)} images to '{output_folder}' folder")
            else:
                st.warning("No matching faces found. Try adjusting the threshold or upload a different photo.")
        else:
            st.info("Upload a photo and click 'Find Similar Faces' to see results")

    # Instructions section
    with st.expander("How to use this app"):
        st.markdown("""
        1. **Install required packages**: 
           ```bash
           pip install ultralytics opencv-python tensorflow streamlit
           ```
        2. **Upload a photo** of the person you want to find
        3. **Set the model path** to your trained face recognition model
        4. **Specify the database folder** containing images to search through
        5. **Adjust the similarity threshold**:
           - Lower values (0.5-0.65): More strict matching, fewer results
           - Higher values (0.65-0.8): More lenient matching, more results (but possibly incorrect matches)
        6. **Adjust face detection confidence**:
           - Higher values: More confident detections, but might miss some faces
           - Lower values: More detections, but might include false positives
        7. Click **"Find Similar Faces"** to search for matches
        8. Use **"Save Results to Folder"** to save all matching face images
        
        **Tips for best results:**
        - Use clear, front-facing photos
        - Ensure good lighting in the images
        - The model works best with consistent image quality
        - Start with a threshold around 0.65-0.7 and adjust as needed
        """)

if __name__ == "__main__":
    main()