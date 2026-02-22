# Gravat Detection and Volume Estimation

## Overview

This project provides a comprehensive solution for detecting and estimating the volume of "gravat" (debris or rubble) in images, a common challenge in construction and industrial sites. The system leverages advanced computer vision techniques, including semantic segmentation with pre-trained models and unsupervised clustering, to accurately identify and isolate debris from complex backgrounds. The ultimate goal is to provide a reliable method for quantifying debris volume, which is crucial for resource management, site safety, and operational efficiency.

##  Motivation and Importance

Manual estimation of debris volume is a time-consuming, labor-intensive, and often inaccurate process. Automating this task offers significant benefits, including:

*   **Improved Accuracy:** Computer vision models can provide more consistent and precise volume estimations compared to human observation.
*   **Enhanced Safety:** Automated monitoring of debris can help identify potential hazards on construction sites, contributing to a safer working environment.
*   **Optimized Resource Management:** Accurate volume data enables better planning for debris removal and disposal, leading to cost savings and more efficient resource allocation.
*   **Data-Driven Insights:** The collected data can be used for long-term analysis, helping to optimize workflows and improve project management.

## Key Features

*   **Multiple Detection Approaches:** The project explores and implements several methods for debris detection, including:
    *   **DeepLabV3:** A state-of-the-art semantic segmentation model pre-trained on the COCO dataset, capable of identifying a wide range of objects and materials.
    *   **SegFormer:** A powerful and efficient Transformer-based model for semantic segmentation, fine-tuned on the ADE20K dataset.
    *   **Unsupervised K-Means Clustering:** A data-driven approach to segmenting images based on color and texture, useful when pre-trained models do not cover the specific type of debris.
*   **Volume Estimation:** The system includes a detailed methodology for estimating the surface area and volume of the detected debris, taking into account real-world reference measurements.
*   **Semantic Similarity for Zero-Shot Detection:** A novel approach that combines image segmentation with language models (SentenceTransformers) to identify debris without requiring a model pre-trained on a specific "gravat" class. This allows for flexible, zero-shot detection of various materials.

## Methodologies Explored

This project investigates two primary methodologies for gravat detection and analysis:

### Approach 1: Segmentation and Volume Calculation (Notebook: `gravat-detection-volume-calculation_amghar_hind_final.ipynb`)

This approach focuses on segmenting debris from images and subsequently calculating its volume. It explores different segmentation techniques:

1.  **DeepLabV3 Segmentation:**
    *   **Process:** Utilizes a pre-trained DeepLabV3 ResNet-101 model (trained on COCO dataset) to perform semantic segmentation on input images. The model identifies various object classes, and a specific class ID corresponding to debris-like material is used to generate a binary mask.
    *   **Example:** The notebook demonstrates loading an image, preprocessing it, running inference with DeepLabV3, and generating a mask for a target class (e.g., class ID 12 for "gravat-like" material).

2.  **K-Means Clustering for Segmentation:**
    *   **Process:** Applies K-Means clustering directly to the image pixel data to group similar colors/textures into clusters. One of these clusters is then identified as representing the debris.
    *   **Example:** The notebook illustrates loading an image, reshaping its pixels, applying K-Means (e.g., with k=3), and visualizing the resulting binary mask for a chosen cluster.

3.  **Volume Calculation:**
    *   **Process:** After obtaining a binary mask of the debris, the system calculates the pixel area of the detected region. By providing a known real-world reference measurement (e.g., the width of an object in the image), the pixel area is converted into a real-world surface area (e.g., in square centimeters). Volume can then be estimated based on this surface area and an assumed average height.
    *   **Example:** The notebook includes functions to calculate surface area from a mask and a reference width, demonstrating how to derive quantitative measurements from the segmentation results.

### Approach 2: Zero-Shot Detection with SegFormer and Semantic Similarity (Notebook: `gravat-segformer-b0-finetuned-ade-512-512.ipynb`)

This innovative approach combines a powerful segmentation model with a language model to identify debris semantically, without explicit training on a "gravat" class:

1.  **SegFormer Segmentation:**
    *   **Process:** An image is processed by a pre-trained SegFormer model (e.g., `nvidia/segformer-b0-finetuned-ade-512-512`). This model outputs segmentation masks for all detected objects and their corresponding labels.
    *   **Example:** The notebook shows how to load an image, use the SegFormer feature extractor and model, perform inference, and obtain predicted class labels for each pixel.

2.  **Semantic Identification with SentenceTransformers:**
    *   **Process:** The labels generated by SegFormer are encoded into high-dimensional vectors using a SentenceTransformer model. Simultaneously, a target query (e.g., "debris pile" or "rubble") is also encoded. Cosine similarity is then calculated between the query vector and all label vectors. The object label with the highest similarity score is identified as the target debris.
    *   **Example:** The notebook demonstrates how to map the segmentation result to a binary mask for a semantically identified class (e.g., class ID 124 for "gravat-like" material in ADE20K dataset) and visualize the result.

## Technology Stack

*   **Core Libraries:** PyTorch, NumPy, Pandas, OpenCV, Matplotlib, PIL
*   **Semantic Segmentation Models:** DeepLabV3 (from `torchvision`), SegFormer (from Hugging Face `transformers`)
*   **Semantic Similarity:** SentenceTransformer (from Hugging Face `sentence-transformers`)
*   **Development Environment:** Jupyter Notebook

## Project Structure and Files

```
gravat-detection/
├── README.md                                                     # This file
├── gravat-detection-volume-calculation_amghar_hind_final.ipynb   # Jupyter notebook implementing DeepLabV3, K-Means, and volume calculation
└── gravat-segformer-b0-finetuned-ade-512-512.ipynb             # Jupyter notebook implementing SegFormer with semantic similarity for zero-shot detection
```

## Getting Started

To run this project, you will need a Python environment with the necessary libraries installed. You can install the required packages using pip:

```bash
pip install torch torchvision numpy pandas opencv-python matplotlib pillow transformers sentence-transformers
```

Then, you can open and run the Jupyter notebooks to see the implementation and results for each approach.

## Results and Performance

*   **Accurate Debris Segmentation:** Both approaches are capable of generating precise masks that effectively isolate debris from the background, with varying degrees of accuracy depending on the model and scene complexity.
*   **Reliable Volume Estimation:** The volume estimation provides a good approximation of the amount of debris, which can be further improved with more sophisticated 3D data.
*   **Flexible Zero-Shot Detection:** The semantic similarity approach demonstrates the ability to detect new types of debris without retraining the vision model, offering a highly adaptable solution.
<img width="800" height="290" alt="image" src="https://github.com/user-attachments/assets/a68140f8-85bb-4dd2-ada0-58537fce8f65" />

## Limitations and Future Work

*   **2D to 3D Assumption:** The current volume estimation is based on a 2D surface area and an assumed height, which is an approximation. Future work could involve using stereo cameras or other 3D sensors to capture more accurate depth information for true volumetric calculations.
*   **Model Generalization:** The performance of the pre-trained models may vary depending on the specific type of debris and the complexity of the scene. Fine-tuning the models on a custom dataset of "gravat" images would likely improve performance.
*   **Lighting and Environmental Conditions:** The accuracy of the segmentation can be affected by challenging lighting conditions, shadows, and occlusions. Advanced data augmentation and more robust models could help mitigate these issues.
*   **Real-Time Processing:** For real-time applications, the processing pipeline would need to be optimized for speed, potentially by using smaller models or hardware acceleration.

## Acknowledgement

I would like to express my sincere gratitude to my supervisor, **EL HABIB Ben Lahmar**, for his support and valuable guidance throughout this project.

