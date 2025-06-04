# test.py

import pytest
import numpy as np
import os
from PIL import Image

# Ensure model.py and its dependencies (like ONNX runtime) are available
# Also assumes convert_to_onnx.py has been run to create model.onnx
try:
    from model import preprocess_image, OnnxModel
except ImportError as e:
    print(f"Error importing from model.py: {e}")
    print("Please ensure model.py is in the same directory and dependencies are installed.")
    # Pytest will likely fail before this, but good practice to check
    pytest.skip("Skipping tests due to missing model.py or dependencies", allow_module_level=True)

# Define paths to test assets
ONNX_MODEL_PATH = "model.onnx"
IMAGE_TENCH = "n01440764_tench.jpeg"       # Expected class ID 0
IMAGE_TURTLE = "n01667114_mud_turtle.JPEG" # Expected class ID 35
IMAGE_INVALID = "non_existent_image.jpg"
IMAGE_DIR = "." # Assuming images are in the same directory

# Expected dimensions after preprocessing
EXPECTED_SHAPE = (1, 3, 224, 224)
EXPECTED_CLASSES = 1000

# --- Fixtures --- (Optional, but good practice for setup/teardown)

@pytest.fixture(scope="module")
def onnx_model():
    """Fixture to load the ONNX model once per module."""
    if not os.path.exists(ONNX_MODEL_PATH):
        pytest.skip(f"Skipping ONNX model tests: {ONNX_MODEL_PATH} not found. Run convert_to_onnx.py first.")
    try:
        model = OnnxModel(model_path=ONNX_MODEL_PATH)
        return model
    except Exception as e:
        pytest.fail(f"Failed to load ONNX model fixture: {e}")

# --- Test Cases --- 

# 1. Test Image Preprocessing
@pytest.mark.parametrize("image_file, exists", [
    (IMAGE_TENCH, True),
    (IMAGE_TURTLE, True),
    (IMAGE_INVALID, False),
])
def test_preprocess_image_existence(image_file, exists):
    """Test if preprocess_image handles existing and non-existing files."""
    image_path = os.path.join(IMAGE_DIR, image_file)
    if not exists:
        # For non-existent files, expect None
        assert preprocess_image(image_path) is None
    else:
        # For existing files, ensure the file actually exists before testing
        if not os.path.exists(image_path):
             pytest.skip(f"Skipping test: Sample image {image_path} not found.")
        # Expect a numpy array if file exists and is processable
        result = preprocess_image(image_path)
        assert isinstance(result, np.ndarray)

def test_preprocess_image_output():
    """Test the output shape and type of the preprocessing function."""
    image_path_tench = os.path.join(IMAGE_DIR, IMAGE_TENCH)
    if not os.path.exists(image_path_tench):
        pytest.skip(f"Skipping test: Sample image {image_path_tench} not found.")
        
    preprocessed = preprocess_image(image_path_tench)
    assert preprocessed is not None
    assert preprocessed.shape == EXPECTED_SHAPE
    assert preprocessed.dtype == np.float32 # PyTorch tensors usually convert to float32

# 2. Test ONNX Model Loading
def test_onnx_model_loading(onnx_model):
    """Test if the OnnxModel class loads the model correctly (using fixture)."""
    assert onnx_model is not None
    assert onnx_model.session is not None
    assert onnx_model.input_name is not None
    assert onnx_model.output_name is not None

# 3. Test ONNX Model Prediction
@pytest.mark.parametrize("image_file, expected_class_id", [
    (IMAGE_TENCH, 0),
    (IMAGE_TURTLE, 35),
])
def test_onnx_prediction(onnx_model, image_file, expected_class_id):
    """Test the full prediction pipeline: preprocess -> predict -> check class ID."""
    image_path = os.path.join(IMAGE_DIR, image_file)
    if not os.path.exists(image_path):
        pytest.skip(f"Skipping test: Sample image {image_path} not found.")

    # Preprocess
    preprocessed_data = preprocess_image(image_path)
    assert preprocessed_data is not None, f"Preprocessing failed for {image_file}"
    assert preprocessed_data.shape == EXPECTED_SHAPE

    # Predict
    predictions = onnx_model.predict(preprocessed_data)
    assert predictions is not None, f"Prediction failed for {image_file}"
    
    # Check output shape
    assert predictions.shape == (1, EXPECTED_CLASSES), "Prediction output shape mismatch"

    # Check predicted class ID
    predicted_class_id = np.argmax(predictions, axis=1)[0]
    print(f"Image: {image_file}, Predicted ID: {predicted_class_id}, Expected ID: {expected_class_id}")
    assert predicted_class_id == expected_class_id, f"Incorrect prediction for {image_file}"

# --- Optional: Add more tests --- 
# - Test with incorrectly shaped input to predict method
# - Test edge cases for preprocessing (e.g., different image modes if not handled)
# - Test performance/latency (might be harder in a simple test script)

# To run these tests: 
# 1. Make sure you have run `convert_to_onnx.py` successfully.
# 2. Install pytest: `pip install pytest`
# 3. Run pytest from the terminal in this directory: `pytest`

