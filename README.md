# Future Frame Synthesis Using Deep Learning
### Required Libraries

pip install torch torchvision matplotlib numpy opencv-python tqdm pillow

### Project Structure

* **`config.py`**
    * Stores global hyperparameters.
    * Contains the `VIDEO_CONFIG` dictionary which defines preprocessing parameters for each video dataset.
 
* **`dataset.py`**
    * Handles data preparation.
    * Defines the `FutureFramePredictionDataset` class.

* **`models.py`**
    * Defines the main neural network architecture: `FutureFramePredictionModel`.
 
* **`layers.py`**
    * Defines the `CrossConvolutionalLayer`.

* **`train.py`**
    * Includes argument parsing, the training loop, loss calculation, model checkpointing, and periodic evaluation.
  
* **`utils.py`**
    * Provides helper functions for reproducibility, loss calculation, and visualization tools.
 
### Dataset
https://drive.google.com/drive/folders/1vNFT8eiZIWB638lDaBq5s37-EcAEuPeL?usp=sharing
