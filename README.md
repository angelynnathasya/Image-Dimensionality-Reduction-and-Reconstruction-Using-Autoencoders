# Image Dimensionality Reduction with Convolutional Autoencoders

A deep learning project using convolutional autoencoders to compress 28×28 grayscale images of cars and planes from 784 pixels to a 128-dimensional latent representation. Compares baseline and modified architectures using reconstruction quality, visual inspection, and Structural Similarity Index (SSIM).

---

## Project Overview

This project explores image dimensionality reduction using an autoencoder on two classes from the Overhead-MNIST dataset: **Car** and **Plane**.

An autoencoder learns to encode an input image into a lower-dimensional latent representation and then decode that representation to reconstruct the original image. The objective is to reduce the input dimensionality from 784 pixels to 128 latent features while retaining as much structural information as possible.

The notebook implements and compares:
1. **Baseline Autoencoder**
2. **Modified Autoencoder**
3. **Modified Autoencoder with manual hyperparameter experiments**

The reconstruction quality is evaluated using the Structural Similarity Index (SSIM), along with visual comparisons between original and reconstructed images.

## Objectives

- Load, preprocess, scale, and visualize the image dataset.
- Split the dataset into training, validation, and test subsets.
- Build a baseline autoencoder with a 128-dimensional latent representation.
- Modify the architecture to improve image reconstruction.
- Experiment with learning rate, batch size, and training epochs.
- Compare reconstruction quality using SSIM.
- Analyze the trade-off between model complexity and reconstruction performance.

## Dataset

The notebook uses two image classes:
- **Car:** 8,012 images
- **Plane:** 8,012 images
- **Total:** 16,024 images

Each image is converted to grayscale and resized to **28 × 28 pixels**. Pixel values are normalized from the range 0–255 to 0–1.

The input image therefore has 784 pixel values, while the encoder maps it to a latent vector of 128 dimensions. This retains approximately **16.33%** of the original dimensionality, corresponding to an **83.67% dimensionality reduction**.

The dataset is based on the Overhead-MNIST dataset. The dataset source and usage terms should be checked before redistributing the image files.

## Data Preparation

The notebook combines images read from the supplied `train` and `test` directories, then creates a new stratified split:

| Subset | Images | Approximate proportion |
|---|---:|---:|
| Training | 12,818 | 80% |
| Validation | 1,603 | 10% |
| Test | 1,603 | 10% |
| **Total** | **16,024** | **100%** |

The split uses `stratify` to preserve the class proportions across subsets.

**Data-split note:** The original `train` and `test` folders are combined before the 80:10:10 split. Therefore, the resulting test subset is a newly created split, not the original dataset test folder. This procedure should be clearly documented. For a stricter final evaluation, preserve the official test folder and split only the original training data into training and validation subsets if that is compatible with the assignment requirements.

## Methodology

### 1. Preprocessing
- Load images from the Car and Plane classes.
- Convert images to grayscale.
- Resize each image to 28 × 28.
- Normalize pixel values to `[0, 1]`.
- Add a channel dimension so each input has shape `28 × 28 × 1`.
- Split the data into training, validation, and test subsets.

### 2. Baseline Autoencoder
The baseline model uses convolutional feature extraction, pooling, flattening, and a dense latent layer. The decoder expands the latent representation and reconstructs an image of shape `28 × 28 × 1`.

- Input: `28 × 28 × 1`
- Latent representation: `128` dimensions
- Output: `28 × 28 × 1`
- Optimizer: Adam
- Loss: Binary cross-entropy
- Training: up to 50 epochs, batch size 32

### 3. Modified Autoencoder
The modified architecture adds convolutional layers to the encoder and decoder to increase the model's capacity to learn image features at different levels.

- Input: `28 × 28 × 1`
- Latent representation: `128` dimensions
- Output: `28 × 28 × 1`
- Optimizer: Adam with learning rate `0.0005`
- Loss: Binary cross-entropy
- Early stopping based on validation loss
- Maximum training: 100 epochs, batch size 16

The latent dimension remains 128 in both architectures, allowing reconstruction quality to be compared while keeping the representation size fixed.

### 4. Hyperparameter Experiments
The notebook tests the following configurations with the modified architecture:

| Trial | Learning rate | Batch size | Maximum epochs |
|---|---:|---:|---:|
| Trial 1 | 0.001 | 32 | 50 |
| Trial 2 | 0.0005 | 32 | 50 |
| Trial 3 | 0.0005 | 16 | 100 |

Early stopping monitors validation loss and restores the best weights. The experiments explore how optimization settings affect reconstruction quality.

**Evaluation note:** In the current notebook, SSIM for these tuning trials is calculated on the test subset and is then used to choose the best trial. This makes the test subset part of model selection, so the resulting tuning score should be treated as exploratory rather than an unbiased final test result. Before presenting the selected tuned model as a final result, select the configuration using validation SSIM, then evaluate the selected model on the test set only after selection.

## Evaluation Metric: SSIM

The **Structural Similarity Index (SSIM)** measures structural similarity between an original image and its reconstruction. It compares aspects of image luminance, contrast, and structure. In general, a higher SSIM indicates greater structural similarity.

The notebook calculates SSIM for each test image and reports the mean score for each model.

## Results

The notebook's current outputs report the following average SSIM values:

| Model | Average SSIM |
|---|---:|
| Baseline Autoencoder | 0.6850 |
| Modified Autoencoder | 0.8244 |
| Modified Autoencoder — Trial 2 | 0.8344 |

### Interpretation

- The Modified Autoencoder's reported average SSIM is higher than the baseline score, suggesting improved reconstruction quality in the current experiment.
- Trial 2 has the highest reported tuning SSIM among the three trials.
- The modified architecture adds convolutional layers, which may help the encoder learn richer image features and the decoder reconstruct image structure more effectively.
- These scores should be interpreted in light of the tuning protocol: because trial selection currently uses test-set SSIM, the tuned score is not an independent final test estimate. Re-run tuning using validation SSIM before making a definitive claim about the best tuned model.

## Requirements

The notebook uses:
- Python
- NumPy
- pandas
- Matplotlib
- Pillow
- scikit-learn
- TensorFlow / Keras
- scikit-image
- Jupyter Notebook



---

*Academic project on image dimensionality reduction and reconstruction using deep learning.*
