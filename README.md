# Automated Gravat Detection with Vision and Language Models

This project presents an automated solution for detecting debris in images, primarily for construction sites, by leveraging a combination of computer vision and semantic similarity techniques. The goal is to efficiently identify piles of debris, extract their masks for analysis, and improve safety and resource management.

## Overview

Traditional object detection models require extensive training on specific object classes. This project bypasses that by using a two-stage approach:

1.  **Detect Everything:** An image segmentation model (SegFormer) first identifies all distinct objects and regions in an image.
2.  **Identify Semantically:** A language model (SentenceTransformer) then determines which of those detected objects is semantically closest to the concept of "debris" or "rubble."

This method is highly flexible and can be adapted to detect various concepts without retraining the vision model.

## Technology Stack

-   **Image Segmentation:** **SegFormer** via the Hugging Face `transformers` library.
-   **Semantic Similarity:** **SentenceTransformer** to compute embeddings for object labels and target concepts.
-   **Core Libraries:** PyTorch, PIL, NumPy.

## How It Works

1.  An image of a construction site is provided as input.
2.  The SegFormer model processes the image and outputs segmentation masks for all detected objects, along with their corresponding labels (e.g., "rock," "sand," "ground").
3.  The SentenceTransformer model encodes both the detected labels and a target query (e.g., "debris pile") into high-dimensional vectors.
4.  Cosine similarity is calculated between the query vector and all label vectors.
5.  The label with the highest similarity score is identified as debris, and its corresponding mask is extracted for further analysis, such as volume calculation.

## Acknowledgement

I would like to express my sincere gratitude to my supervisor, **EL HABIB Ben Lahmar**, for his support and valuable guidance throughout this project.
