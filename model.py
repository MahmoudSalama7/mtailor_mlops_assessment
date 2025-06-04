# model.py

import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os

class OnnxModel:
    """Handles loading the ONNX model and running predictions."""

    def __init__(self, model_path="model.onnx"):
        """Initializes the ONNX model loader.

        Args:
            model_path (str): Path to the .onnx model file.
        """
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self._load_model()

    def _load_model(self):
        """Loads the ONNX model into an inference session."""
        if not os.path.exists(self.model_path):
            print(f"Error: ONNX model file not found at {self.model_path}")
            print("Please run convert_to_onnx.py first to generate the model.")
            # Raise an error or handle appropriately
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        try:
            print(f"Loading ONNX model from {self.model_path}...")
            # Load the model using ONNX Runtime
            # Consider specifying providers like ["CPUExecutionProvider"] if needed
            self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            
            # Get input and output names from the model
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            print(f"ONNX model loaded successfully.")
            print(f"  Input Name: {self.input_name}")
            print(f"  Output Name: {self.output_name}")

        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            # Raise an error or handle appropriately
            raise RuntimeError(f"Failed to load ONNX model: {e}")

    def predict(self, preprocessed_image: np.ndarray) -> np.ndarray:
        """Runs inference on the preprocessed image.

        Args:
            preprocessed_image (np.ndarray): The image data, preprocessed and ready for the model.
                                             Expected shape: (batch_size, channels, height, width)

        Returns:
            np.ndarray: The model's output probabilities.
        """
        if self.session is None:
            print("Error: ONNX session not initialized. Load the model first.")
            return None # Or raise an error

        if self.input_name is None:
            print("Error: Model input name not identified.")
            return None # Or raise an error

        try:
            # Run inference
            # Input must be a dictionary mapping input names to NumPy arrays
            input_feed = {self.input_name: preprocessed_image}
            outputs = self.session.run([self.output_name], input_feed)
            # outputs is a list, get the first element which corresponds to self.output_name
            return outputs[0]
        except Exception as e:
            print(f"Error during ONNX inference: {e}")
            return None # Or raise an error

def preprocess_image(image_path: str) -> np.ndarray:
    """Loads an image, preprocesses it, and returns it as a NumPy array.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        np.ndarray: The preprocessed image as a NumPy array (batch_size, C, H, W).
                    Returns None if preprocessing fails.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    try:
        # Load the image using PIL
        img = Image.open(image_path)

        # Convert to RGB if necessary (e.g., if it's RGBA or Grayscale)
        if img.mode != "RGB":
            print(f"Converting image from {img.mode} to RGB.")
            img = img.convert("RGB")

        # Define the same preprocessing steps used in PyTorch model/training
        # Resize, CenterCrop, ToTensor, Normalize
        preprocess = transforms.Compose([
            transforms.Resize(256),             # Resize smaller edge to 256
            transforms.CenterCrop(224),         # Crop center 224x224
            transforms.ToTensor(),              # Convert to tensor (0-1 range) and C, H, W format
            transforms.Normalize(               # Normalize with ImageNet stats
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Apply preprocessing
        img_tensor = preprocess(img)

        # Add batch dimension (unsqueeze) and convert to NumPy array
        # ONNX Runtime expects NumPy arrays as input
        img_numpy = img_tensor.unsqueeze(0).numpy()

        return img_numpy

    except Exception as e:
        print(f"Error during image preprocessing ({image_path}): {e}")
        return None

# Example Usage (can be run standalone for basic testing)
if __name__ == "__main__":
    print("--- Running model.py standalone example ---")
    
    # Define paths for example
    TEST_IMAGE_PATH = "n01440764_tench.jpeg" # Tench (class ID 0)
    # TEST_IMAGE_PATH = "n01667114_mud_turtle.JPEG" # Mud Turtle (class ID 35)
    ONNX_PATH = "model.onnx"

    # 1. Preprocess the image
    print(f"Preprocessing image: {TEST_IMAGE_PATH}")
    preprocessed_data = preprocess_image(TEST_IMAGE_PATH)

    if preprocessed_data is not None:
        print(f"Image preprocessed successfully. Shape: {preprocessed_data.shape}, Type: {preprocessed_data.dtype}")
        
        # 2. Load the ONNX model
        try:
            onnx_model = OnnxModel(model_path=ONNX_PATH)
            
            # 3. Run prediction
            print("Running prediction...")
            predictions = onnx_model.predict(preprocessed_data)

            if predictions is not None:
                print(f"Prediction output shape: {predictions.shape}")
                # The output is typically logits or probabilities for each class
                # Find the class with the highest probability
                predicted_class_id = np.argmax(predictions, axis=1)[0]
                print(f"Predicted Class ID: {predicted_class_id}")
                # You might want to load class labels from a file to show the class name
            else:
                print("Prediction failed.")

        except (FileNotFoundError, RuntimeError) as e:
            print(f"Could not run prediction: {e}")
        except Exception as e:
             print(f"An unexpected error occurred: {e}")

    else:
        print("Image preprocessing failed. Cannot run prediction.")

    print("--- End of model.py standalone example ---")

