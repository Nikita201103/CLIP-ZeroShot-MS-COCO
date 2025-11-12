This project demonstrates the implementation of text-to-image and image-to-text retrieval, along with zero-shot classification capabilities, using the Contrastive Language-Image Pre-training (CLIP) model. The MS-COCO 2017 validation dataset is used for both training/support and evaluation.

Project Overview
The goal of this project is to explore CLIP's effectiveness in various retrieval and classification tasks without explicit training on target labels. The process involves:

Loading and Encoding Data: Processing image and text data from the MS-COCO dataset, extracting features using a pre-trained CLIP model.
Text-to-Image Retrieval: Given a caption, retrieving the most relevant images.
Image-to-Text Retrieval: Given an image, retrieving the most relevant captions.
Zero-Shot Classification (Caption to Image Category): Given a test caption, predicting its categories by finding the nearest images in a 'training' set and aggregating their ground-truth categories.
Zero-Shot Classification (Image to Caption Category): Given a test image, predicting its categories by finding the nearest captions in a 'training' set and aggregating their ground-truth categories.
Tasks Implemented
Task 1: Caption → Image Retrieval and Zero-Shot Classification
This task focuses on retrieving images based on a textual query (caption). For each query, the top-k most similar images are retrieved, and their combined ground-truth COCO categories are used as a prediction. Metrics like At-least-k Accuracy, Jaccard, Precision, Recall, and F1-score are used for evaluation.

Example Output Snippet:

Caption: A black Honda motorcycle parked in front of a garage.
Ground Truth Categories: ['motorcycle']
Predicted Categories (from top-5 images): ['motorcycle', 'car', 'person']
Key Metrics:

At-least-1 Accuracy: 1.0
Jaccard (micro): 0.45
Precision (micro): 0.49
Recall (micro): 0.86
F1 (micro): 0.62
Task 2: Image → Caption Retrieval
This task evaluates CLIP's ability to find relevant captions for a given image. For a test image, the system retrieves the top-k most similar captions. The ground-truth captions associated with the image are compared against the retrieved captions. Metrics include Recall@k and multi-label metrics.

Example Output Snippet:

Ground Truth Captions: ['A man is in a kitchen making pizzas.', 'Man in apron standing on front of oven with pans and bakeware', ...]
Top-k Retrieved Captions: ['An old fashioned kitchen is set up for display.', 'People in a kitchen with cooking attire on', ...]
Key Metrics:

Recall@1: 0.515
Recall@5: 0.757
Micro Precision: 0.34
Micro Recall: 0.34
Micro F1: 0.34
Task 3: Test Captions → Nearest Train Image (Zero-Shot Classification)
In this task, a subset of MS-COCO images forms a 'training' set, and another subset forms a 'test' set. For each test caption, the system finds the top-k nearest images from the 'training' set. The union of categories from these top-k images is used to predict the categories for the test caption's ground-truth image.

Example Output Snippet:

Test Caption: An office cubicle with four different types of computers.
Ground-truth Labels of test image: ['potted plant', 'sports ball', 'tv', 'laptop', 'mouse', 'keyboard', 'bottle', 'cup', 'book', 'vase', 'backpack', 'chair']
Predicted Labels (union of top-5 images): ['couch', 'mouse', 'bottle', 'tv', 'chair', 'book', 'cup', 'potted plant', 'laptop', 'keyboard', 'remote']
Key Metrics:

At-least-1 Accuracy: 1.0
Jaccard (micro): 0.29
Precision (micro): 0.31
Recall (micro): 0.80
F1 (micro): 0.45
Task 4: Test Image → Nearest Train Caption (Zero-Shot Classification)
Similar to Task 3, but in reverse. For each test image, the system finds the top-k nearest captions from a 'training' set of captions. The ground-truth categories associated with these top-k captions are aggregated to predict the categories of the test image.

Example Output Snippet:

Test Image (ID: 277584) GT: ['cat', 'bench']
Predicted Labels (union of top-5 captions): ['potted plant', 'cat']
Key Metrics:

At-least-1 Accuracy: 0.98
Jaccard (micro): 0.32
Precision (micro): 0.35
Recall (micro): 0.79
F1 (micro): 0.48
Setup and Usage
To run this notebook, you will need:

MS-COCO 2017 Validation Set: Download the images (val2017) and annotations (captions_val2017.json, instances_val2017.json). Adjust the img_dir, ann_file_caps, and ann_file_cats paths accordingly.
Hugging Face Transformers Library: For CLIP model and processor.
Scikit-learn: For various multi-label classification metrics.
PyTorch: For tensor operations and GPU acceleration.
Clone this repository and run the Jupyter Notebook (.ipynb) cells sequentially.

Dependencies
Install the required libraries using pip:

pip install torch transformers scikit-learn matplotlib numpy pillow tqdm
Conclusion
This project highlights CLIP's powerful zero-shot capabilities for cross-modal retrieval and classification tasks. The results demonstrate its ability to generalize to unseen categories by leveraging its understanding of both images and text.
