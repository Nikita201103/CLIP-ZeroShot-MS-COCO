# 🧠 CLIP-Based Multimodal Retrieval and Zero-Shot Classification

This project demonstrates the implementation of **text-to-image** and **image-to-text retrieval**, along with **zero-shot classification** capabilities, using the **Contrastive Language-Image Pre-training (CLIP)** model. The **MS-COCO 2017 validation dataset** is used for both feature extraction and evaluation.

---

## 📘 Project Overview

The goal of this project is to explore **CLIP's effectiveness** in various retrieval and classification tasks **without explicit training on target labels**.  

### Workflow Summary
1. **Loading and Encoding Data:** Processing image and text data from the MS-COCO dataset, and extracting features using a pre-trained CLIP model.  
2. **Text-to-Image Retrieval:** Given a caption, retrieve the most relevant images.  
3. **Image-to-Text Retrieval:** Given an image, retrieve the most relevant captions.  
4. **Zero-Shot Classification (Caption → Image Category):** Predict the category of a test caption using the nearest images.  
5. **Zero-Shot Classification (Image → Caption Category):** Predict the category of a test image using the nearest captions.

---

## 🎯 Tasks Implemented

### **Task 1: Caption → Image Retrieval and Zero-Shot Classification**
This task retrieves images based on a textual query (caption). For each caption, the **top-k most similar images** are retrieved, and their combined ground-truth categories are used as predictions. Evaluation metrics include *At-least-k Accuracy*, *Jaccard Index*, *Precision*, *Recall*, and *F1-score*.

**Example Output:**
```
Caption: A black Honda motorcycle parked in front of a garage.
Ground Truth Categories: ['motorcycle']
Predicted Categories (from top-5 images): ['motorcycle', 'car', 'person']
```

**Key Metrics:**
| Metric | Score |
|:--------|:------:|
| At-least-1 Accuracy | 1.0 |
| Jaccard (micro) | 0.45 |
| Precision (micro) | 0.49 |
| Recall (micro) | 0.86 |
| F1 (micro) | 0.62 |

---

### **Task 2: Image → Caption Retrieval**
Evaluates CLIP's ability to retrieve relevant captions for a given image. Metrics include *Recall@k* and *multi-label metrics*.

**Example Output:**
```
Ground Truth Captions: ['A man is in a kitchen making pizzas.', 'Man in apron standing in front of oven...', ...]
Top-k Retrieved Captions: ['An old fashioned kitchen is set up for display.', 'People in a kitchen with cooking attire...', ...]
```

**Key Metrics:**
| Metric | Score |
|:--------|:------:|
| Recall@1 | 0.515 |
| Recall@5 | 0.757 |
| Micro Precision | 0.34 |
| Micro Recall | 0.34 |
| Micro F1 | 0.34 |

---

### **Task 3: Test Captions → Nearest Train Image (Zero-Shot Classification)**
A subset of images forms the “training” set, and another subset forms the “test” set. Each test caption finds the top-k nearest training images; their combined categories are used as predictions.

**Example Output:**
```
Test Caption: An office cubicle with four different types of computers.
Ground-truth Labels: ['potted plant', 'sports ball', 'tv', 'laptop', 'mouse', 'keyboard', 'bottle', 'cup', 'book', 'vase', 'backpack', 'chair']
Predicted Labels: ['couch', 'mouse', 'bottle', 'tv', 'chair', 'book', 'cup', 'potted plant', 'laptop', 'keyboard', 'remote']
```

**Key Metrics:**
| Metric | Score |
|:--------|:------:|
| At-least-1 Accuracy | 1.0 |
| Jaccard (micro) | 0.29 |
| Precision (micro) | 0.31 |
| Recall (micro) | 0.80 |
| F1 (micro) | 0.45 |

---

### **Task 4: Test Image → Nearest Train Caption (Zero-Shot Classification)**
This task mirrors Task 3 but in reverse. Each test image retrieves the top-k nearest captions from a training caption set, and their associated categories form the predictions.

**Example Output:**
```
Test Image (ID: 277584)
Ground Truth: ['cat', 'bench']
Predicted Labels: ['potted plant', 'cat']
```

**Key Metrics:**
| Metric | Score |
|:--------|:------:|
| At-least-1 Accuracy | 0.98 |
| Jaccard (micro) | 0.32 |
| Precision (micro) | 0.35 |
| Recall (micro) | 0.79 |
| F1 (micro) | 0.48 |

---

## ⚙️ Setup and Usage

### **Dataset**
Download the **MS-COCO 2017 Validation Set**:
- `val2017` images
- `captions_val2017.json`
- `instances_val2017.json`

Adjust the paths in the notebook:
```python
img_dir = "path/to/val2017"
ann_file_caps = "path/to/captions_val2017.json"
ann_file_cats = "path/to/instances_val2017.json"
```

---

### **Dependencies**
Install the required Python libraries:
```bash
pip install torch transformers scikit-learn matplotlib numpy pillow tqdm
```

---

### **Running the Notebook**
1. Clone this repository.  
2. Open the `.ipynb` notebook.  
3. Run all cells sequentially.

---

## 🧪 Results Summary
- CLIP effectively performs **cross-modal retrieval** without fine-tuning.  
- The model demonstrates strong **zero-shot classification** capabilities.  
- Text and image representations align semantically, allowing accurate retrievals even for unseen categories.

---

## 🧠 Conclusion
This project highlights CLIP’s **powerful zero-shot learning capabilities** for **cross-modal retrieval and classification**. It shows that CLIP can generalize well to **unseen categories** by leveraging its joint understanding of **visual** and **textual** semantics.

---

---

## 🧰 Tech Stack
- **Model:** CLIP (OpenAI)  
- **Frameworks:** PyTorch, Transformers (Hugging Face)  
- **Dataset:** MS COCO 2017 Validation Set  
- **Evaluation:** Scikit-learn metrics
