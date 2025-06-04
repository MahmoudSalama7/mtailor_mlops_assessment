# MLOps Assessment - Image Classification Deployment

This repository contains the code and instructions for deploying a PyTorch image classification model (ResNet18 trained on ImageNet) to Cerebrium using ONNX and Docker, as per the Mtailor MLOps assessment requirements.

Due to environment issues during development, the focus shifted to providing the necessary code files and instructions for you to run and deploy from your local machine.

## Project Structure

```
mtailor_mlops_assessment/
├── convert_to_onnx.py      # Script to convert PyTorch weights to ONNX format
├── model.py                # Contains ONNX model loading and image preprocessing logic
├── test.py                 # Pytest script for local testing of model.py functionalities
├── main.py                 # FastAPI application for Cerebrium deployment
├── test_server.py          # Script to test the deployed Cerebrium endpoint
├── Dockerfile              # Dockerfile for building the deployment image
├── requirements.txt        # Python dependencies
├── pytorch_model.py        # Original PyTorch model definition (required by convert_to_onnx.py)
├── pytorch_model_weights.pth # Downloaded PyTorch model weights
├── model.onnx              # Generated ONNX model file (after running convert_to_onnx.py)
├── n01440764_tench.jpeg    # Sample image 1 (Class ID 0)
├── n01667114_mud_turtle.JPEG # Sample image 2 (Class ID 35)
└── README.md               # This file
```

## Deliverables

1.  **`convert_to_onnx.py`**: Converts the provided PyTorch model weights (`pytorch_model_weights.pth`) into an ONNX model (`model.onnx`). Requires `pytorch_model.py` and the weights file.
2.  **`model.py`**: Contains:
    *   `preprocess_image(image_path)`: Function to load and preprocess an image from a file path according to ImageNet standards.
    *   `OnnxModel(model_path)`: Class to load the `model.onnx` file using ONNX Runtime and perform inference.
3.  **`test.py`**: Contains Pytest tests for `model.py`. It verifies:
    *   Image preprocessing for valid and invalid inputs.
    *   Correct output shape and type from preprocessing.
    *   Successful loading of the ONNX model.
    *   Correct prediction (class ID) for the provided sample images using the ONNX model.
4.  **Cerebrium Deployment Files**:
    *   **`main.py`**: A FastAPI application that loads the ONNX model and provides a `/predict` endpoint accepting base64 encoded images.
    *   **`Dockerfile`**: Defines the Docker image environment, installs dependencies, copies code, and sets up for running the FastAPI app.
    *   **`requirements.txt`**: Lists all necessary Python packages.
5.  **`test_server.py`**: A script to test the *deployed* Cerebrium endpoint. It can:
    *   Send a single image (via `--image_path`) for prediction.
    *   Run preset tests (`--run_preset_tests`) using the sample images to verify deployment correctness.
    *   Requires the deployed endpoint URL (`--api_url`) and your Cerebrium API key (`--api_key`).
6.  **`README.md`**: This file, providing setup and usage instructions.

## Setup and Local Execution

**Prerequisites:**

*   Python 3.9+ installed.
*   Git installed (optional, for cloning if you host this code).
*   Access to download the PyTorch weights file.

**Steps:**

1.  **Create Project Directory:** Create a folder named `mtailor_mlops_assessment` on your local machine.
2.  **Place Files:** Copy all the files provided (including this README, `.py` files, `Dockerfile`, `requirements.txt`) into this directory.
3.  **Download Assets:**
    *   Download the PyTorch weights from the link provided in the assessment and save it as `pytorch_model_weights.pth` in the project directory.
    *   Ensure the sample images (`n01440764_tench.jpeg`, `n01667114_mud_turtle.JPEG`) are present in the directory.
4.  **Set up Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
5.  **Install Dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
6.  **Convert PyTorch Model to ONNX:**
    ```bash
    python convert_to_onnx.py
    ```
    This will generate the `model.onnx` file.
7.  **Run Local Tests (Pytest):**
    ```bash
    pytest test.py -v
    ```
    This tests the `preprocess_image` function and the `OnnxModel` class using the generated `model.onnx` and sample images.
8.  **Run FastAPI App Locally (Optional):**
    To test the `main.py` server locally before deployment:
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```
    You can then send POST requests to `http://localhost:8000/predict` with a JSON body like `{"image_base64": "<your_base64_encoded_image>"}`.

## Cerebrium Deployment (Using Custom Docker Image)

**Prerequisites:**

*   Docker installed and running locally.
*   Cerebrium account ([https://www.cerebrium.ai/](https://www.cerebrium.ai/)).
*   Cerebrium CLI installed and configured (or use their web UI for deployment).
*   `model.onnx` file generated locally (from Step 6 above).

**Steps:**

1.  **Build the Docker Image:**
    Navigate to the project directory in your terminal.
    ```bash
    docker build -t mtailor-mlops-assessment:latest .
    ```
    *Note: Ensure `model.onnx` exists before building.* 
2.  **Push Docker Image to a Registry (e.g., Docker Hub, AWS ECR, GitHub Container Registry):**
    *   Tag your image: `docker tag mtailor-mlops-assessment:latest <your-registry>/mtailor-mlops-assessment:latest`
    *   Log in to your registry: `docker login <your-registry>`
    *   Push the image: `docker push <your-registry>/mtailor-mlops-assessment:latest`
3.  **Deploy on Cerebrium:**
    *   Use the Cerebrium Dashboard or CLI to deploy a new model.
    *   Choose the "Custom Docker Image" deployment option.
    *   Provide the full path to your pushed Docker image (e.g., `<your-registry>/mtailor-mlops-assessment:latest`).
    *   Configure hardware settings (GPU might be needed depending on performance requirements, but ONNX Runtime CPU should work).
    *   Follow Cerebrium's prompts to complete the deployment.
    *   Cerebrium should automatically detect the FastAPI application within the container (running on port 80 as per common practice, though `main.py` uses 8000 locally - Cerebrium usually maps this). Refer to Cerebrium docs if the port needs explicit configuration during deployment.
4.  **Obtain Endpoint URL and API Key:**
    Once deployed, Cerebrium will provide you with:
    *   A unique API endpoint URL.
    *   An API key for authentication.
5.  **Test the Deployed Endpoint:**
    *   **Note:** Once deployed, replace `<YOUR_CEREBRIUM_ENDPOINT_URL>` below with the actual URL provided by Cerebrium.
    Use the `test_server.py` script:
    ```bash
    python test_server.py --api_url <YOUR_CEREBRIUM_ENDPOINT_URL> --api_key <YOUR_CEREBRIUM_API_KEY> --run_preset_tests
    ```
    Replace placeholders with your actual URL and key.
    You can also test with a single image:
    ```bash
    python test_server.py --api_url <YOUR_CEREBRIUM_ENDPOINT_URL> --api_key <YOUR_CEREBRIUM_API_KEY> --image_path path/to/your/image.jpg
    ```

## Important Notes

*   **Error Handling:** The provided scripts include basic error handling. Robust production deployments would require more comprehensive logging and error management.
*   **Preprocessing:** Preprocessing steps in `model.py` (used by `test.py`) and `main.py` (used in the API) exactly match the steps the original PyTorch model was trained with.
*   **Cerebrium Configuration:** Refer to the official Cerebrium documentation for the most up-to-date details on custom Docker deployments, environment variables, health checks, and scaling.

