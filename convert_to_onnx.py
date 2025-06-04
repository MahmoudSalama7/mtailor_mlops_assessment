# convert_to_onnx.py

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# Assuming pytorch_model.py is in the same directory
# Make sure pytorch_model.py defines Classifier and BasicBlock
try:
    from pytorch_model import Classifier, BasicBlock
except ImportError:
    print("Error: Could not import Classifier or BasicBlock from pytorch_model.py.")
    print("Please ensure pytorch_model.py is in the same directory and defines these classes.")
    exit(1)

# Define paths
PYTORCH_WEIGHTS_PATH = "pytorch_model_weights.pth"
ONNX_MODEL_PATH = "model.onnx"
SAMPLE_IMAGE_PATH = "n01440764_tench.jpeg" # Used for dummy input during export

print("--- Starting PyTorch to ONNX Conversion ---")

# Check if weights file exists
if not os.path.exists(PYTORCH_WEIGHTS_PATH):
    print(f"Error: PyTorch weights file not found at {PYTORCH_WEIGHTS_PATH}")
    print("Please download the weights using the link provided in the original README or assessment details.")
    exit(1)

# Load the PyTorch model definition (ResNet18)
# Ensure the model architecture matches the one used for the provided weights
print("Loading PyTorch model definition (ResNet18)...")
# Using BasicBlock and layers [2, 2, 2, 2] corresponds to ResNet18
pytorch_model = Classifier(BasicBlock, [2, 2, 2, 2], num_classes=1000)

# Load the trained weights
print(f"Loading weights from {PYTORCH_WEIGHTS_PATH}...")
# Load weights onto CPU, as GPU might not be available in all environments
# Use map_location for broader compatibility
patch_dict = torch.load(PYTORCH_WEIGHTS_PATH, map_location=torch.device("cpu"))

# Handle potential key mismatches (e.g., if saved with DataParallel or different naming)
# Basic loading attempt first
try:
    pytorch_model.load_state_dict(patch_dict)
    print("Loaded state_dict successfully.")
except RuntimeError as e:
    print(f"Warning: Standard state_dict loading failed: {e}")
    print("This might happen if the model was saved using DataParallel or has slightly different layer names.")
    print("Attempting to load with strict=False...")
    try:
        pytorch_model.load_state_dict(patch_dict, strict=False)
        print("Loaded state_dict with strict=False. Model structure might differ slightly from saved weights.")
    except Exception as final_e:
        print(f"Error: Failed to load state_dict even with strict=False: {final_e}")
        print("Please ensure the model definition in pytorch_model.py matches the weights file structure.")
        exit(1)

# Set the model to evaluation mode (important for layers like BatchNorm and Dropout)
pytorch_model.eval()
print("PyTorch model loaded and set to evaluation mode.")

# Create a dummy input tensor with the correct shape (batch_size, channels, height, width)
# The preprocessing steps should match those expected by the model during training/inference
print(f"Creating dummy input tensor (1, 3, 224, 224)...")
# Using a random tensor is sufficient for tracing the graph during export
dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False) # No grad needed for export
print("Dummy input created.")

# Define input and output names for the ONNX model (optional but recommended for clarity)
input_names = ["input_image"]
output_names = ["output_probabilities"]

# Export the model to ONNX format
print(f"Exporting model to ONNX format at {ONNX_MODEL_PATH}...")
try:
    torch.onnx.export(
        pytorch_model,               # Model being exported
        dummy_input,                 # Model input (or a tuple for multiple inputs)
        ONNX_MODEL_PATH,             # Where to save the model (can be a file path or file-like object)
        export_params=True,          # Store the trained parameter weights inside the model file
        opset_version=11,            # The ONNX version to export the model to (11 is a common choice)
        do_constant_folding=True,    # Whether to execute constant folding for optimization
        input_names=input_names,     # The model's input names
        output_names=output_names,   # The model's output names
        dynamic_axes={               # Allow variable batch size
            input_names[0]: {0: "batch_size"},
            output_names[0]: {0: "batch_size"}
        }
    )
    print(f"Model successfully exported to {ONNX_MODEL_PATH}")

except Exception as e:
    print(f"Error during ONNX export: {e}")
    # Consider adding more specific error handling if needed
    exit(1)

print("--- ONNX conversion process completed successfully. ---")

