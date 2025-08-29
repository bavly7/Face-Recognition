# Face Finder App 🔍

A powerful Streamlit application for finding similar faces across a collection of images using deep learning and computer vision techniques.

## Features

- **Face Detection**: Utilizes YOLOv8 for accurate face detection in images
- **Face Recognition**: Employs a custom CNN model with triplet loss for robust face embeddings
- **Similarity Search**: Finds all instances of a person across your image database
- **Interactive Interface**: User-friendly web interface with adjustable parameters
- **Batch Processing**: Processes entire folders of images efficiently
- **Results Export**: Save matched faces for further analysis

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/bavly7/Face-Recognition.git
   cd face-finder-app
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare your data**:
   - Organize your image database in a folder
   - Ensure you have a trained face recognition model (`.h5` file)

2. **Run the application**:
   ```bash
   streamlit run app.py
   ```

3. **Use the interface**:
   - Upload a query photo of the person you want to find
   - Set the path to your trained model
   - Specify the folder containing images to search through
   - Adjust the similarity threshold and confidence settings
   - Click "Find Similar Faces" to see results

## Model Architecture

The application uses a custom CNN with triplet loss for face recognition:

```python
def embedding_model(input_shape=(32, 32, 3), embedding_dim=128):
    # CNN blocks with BatchNormalization and Dropout
    # L2-normalized output embeddings
    # Triplet loss training for robust face recognition
```

## Configuration

### Parameters:

- **Similarity Threshold** (0.5-1.0): Controls matching strictness
  - Lower values = more strict matching
  - Higher values = more lenient matching

- **Face Detection Confidence** (0.1-0.9): Controls detection accuracy
  - Higher values = more confident detections
  - Lower values = more detections (may include false positives)

- **Maximum Results**: Number of results to display (5-50)

## File Structure

```
face-finder-app/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── lfw_embedding_model.h5 # Trained model (not included)
└── Images_folder/         # Database of images to search
```

## How It Works

1. **Face Detection**: YOLOv8 identifies and extracts faces from images
2. **Feature Extraction**: Custom CNN generates embedding vectors for each face
3. **Similarity Calculation**: Compares embeddings using Euclidean distance
4. **Results Ranking**: Returns images sorted by similarity score

## Performance Tips

- Use clear, front-facing photos for best results
- Ensure consistent lighting conditions in your images
- Start with a threshold of 0.65-0.7 and adjust as needed
- The model works best with consistent image quality

## Troubleshooting

1. **Model not loading**: Ensure you have the correct model file path
2. **No faces detected**: Adjust the confidence threshold lower
3. **Too many false positives**: Increase the similarity threshold
4. **Memory issues**: Reduce the maximum results or image sizes

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 face detection model from [akanametov/yolov8-face](https://github.com/akanametov/yolov8-face)
- TensorFlow and Keras for deep learning infrastructure
- Streamlit for the web application framework

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Ensure all dependencies are properly installed
3. Verify your model file is compatible with the application

---

**Note**: This application requires a pre-trained face recognition model. The model should be trained using triplet loss on face datasets like LFW, VGGFace, or custom face collections.