# 🦋 Butterfly Species Classification using EfficientNetV2

## 📌 Project Overview

This project focuses on classifying butterfly species from images using Deep Learning and Transfer Learning techniques. The model is trained on a butterfly image dataset and uses **EfficientNetV2-S** as the backbone architecture to achieve high classification accuracy.

The goal is to automatically identify butterfly species from images and generate predictions for unseen test images.

---

## 🚀 Features

* Butterfly image classification
* Transfer Learning with EfficientNetV2-S
* Data Augmentation
* Label Encoding
* Train/Validation Split
* Mixed Precision Training
* Learning Rate Scheduling
* Model Checkpoint Saving
* Test Set Prediction Generation

---

## 📂 Dataset Structure

```text
butterfly_dataset/
│
├── train/
│   ├── Image_1.jpg
│   ├── Image_2.jpg
│   └── ...
│
├── test/
│   ├── Image_1.jpg
│   ├── Image_2.jpg
│   └── ...
│
├── Training_set.csv
└── Testing_set.csv
```

### Training CSV

| filename    | label            |
| ----------- | ---------------- |
| Image_1.jpg | SOUTHERN DOGFACE |
| Image_2.jpg | ADONIS           |

### Testing CSV

| filename    |
| ----------- |
| Image_1.jpg |
| Image_2.jpg |

---

## 🛠 Technologies Used

* Python
* PyTorch
* TIMM
* Pandas
* NumPy
* Scikit-Learn
* TorchVision
* PIL

---

## 🧠 Model Architecture

The project uses:

**EfficientNetV2-S**

Benefits:

* High accuracy
* Fast training
* Efficient parameter usage
* Strong transfer learning performance

```python
model = timm.create_model(
    'tf_efficientnetv2_s',
    pretrained=True,
    num_classes=75
)
```

---

## 🔄 Data Preprocessing

### Training Transformations

* Resize (384 × 384)
* Random Horizontal Flip
* Random Rotation
* Color Jitter
* Normalization

### Validation Transformations

* Resize
* Normalization

---

## ⚙️ Training Configuration

| Parameter     | Value             |
| ------------- | ----------------- |
| Architecture  | EfficientNetV2-S  |
| Optimizer     | AdamW             |
| Learning Rate | 1e-4              |
| Scheduler     | CosineAnnealingLR |
| Loss Function | CrossEntropyLoss  |
| Batch Size    | 32                |
| Epochs        | 25                |

---

## 📈 Model Performance

### Validation Accuracy

**90.08%**

This indicates strong generalization performance on unseen validation samples.

---

## 💾 Model Saving

```python
torch.save(
    model.state_dict(),
    "best_model.pth"
)
```

---

## 🔮 Prediction

The trained model generates predictions for test images and converts encoded labels back to butterfly species names.

Output format:

```csv
filename,label
Image_1.jpg,MONARCH
Image_2.jpg,ADONIS
```

---

## 📊 Future Improvements

* Ensemble Learning
* Vision Transformers (ViT)
* Hyperparameter Optimization
* Model Quantization
* Deployment with Flask/FastAPI

---

## 👨‍💻 Author

**Thoushin Khaled Omi**

Deep Learning | Computer Vision | Machine Learning Enthusiast

---

## ⭐ If you like this project

Give this repository a star ⭐ and support the project.
