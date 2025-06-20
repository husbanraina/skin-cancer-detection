# 🧬 Skin Cancer Detection using CNN

This is a final year B.Tech project that detects different types of skin cancer using a Convolutional Neural Network (CNN). A simple and interactive interface is built using Streamlit to allow users to upload skin lesion images and get predictions in real time.

---

## 📌 Problem Statement

Skin cancer is among the most common cancers worldwide. Early and accurate detection can significantly increase the chances of successful treatment. This project classifies dermoscopic images into one of the following:

- **Melanoma**
- **Nevus**
- **Seborrheic Keratosis**

---

## 📂 Dataset

- **Source**: ISIC Archive ([link](https://www.isic-archive.com/))
- **Images**: Pre-labeled into 3 classes
- **Folder Format After Split**:

```
skin-cancer-detection/
│
├── app/
│   └── app.py
│
├── data/
│   ├── train/
│   └── val/
│
├── model/
│   └── skin_model.h5
│
├── utils/
│   └── split_data.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 🧠 CNN Architecture

- `Conv2D(32, 3x3)` → ReLU → MaxPooling
- `Conv2D(64, 3x3)` → ReLU → MaxPooling
- `Conv2D(128, 3x3)` → ReLU → MaxPooling
- `Flatten` → `Dropout(0.5)`
- `Dense(128)` → `Dense(3)` → `softmax`

**Training Config**:

- Optimizer: `adam`
- Loss: `categorical_crossentropy`
- Epochs: 10
- Batch Size: 32
- Input Size: 128x128x3

---

## 🌐 Streamlit Web App

- Upload an image
- Get prediction with class probabilities
- Clean and minimal UI

---

## ⚙️ Setup & Run Instructions

### ✅ 1. Clone the Repo

```bash
git clone https://github.com/your-username/skin-cancer-detection.git
cd skin-cancer-detection
```

### ✅ 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### ✅ 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### ✅ 4. (Optional) Split Dataset into Train/Val

```bash
python utils/split_data.py
```

### ✅ 5. Train the CNN Model

```bash
python main.py
```

This will create `model/skin_model.h5`.

### ✅ 6. Run the Web App

```bash
streamlit run app/app.py
```

---

## 📸 App Screenshot

> Replace this with your own app screenshot.

```
![App Screenshot](preview.png)
```

---

## 📦 Requirements

Paste this in your `requirements.txt`:

```
tensorflow==2.19.0
numpy==2.1.3
pillow==11.2.1
streamlit==1.35.0
matplotlib==3.10.3
scikit-learn==1.7.0
pandas==2.3.0
protobuf==5.29.5
h5py==3.14.0
```

---

## 👨‍💻 Developed By

**TANISH (210316), MEHVISH NABI (210361), TAJAMUL HUDDA (210364), ANAMUL HAQ WAR (210342)**  
Final Year B.Tech Student – 2021-2025  
India  
Email: (add your email here)

---

## 📜 License

This project is intended for academic use only. Dataset and image sources belong to their respective owners.

---

## 💡 Extras

- Easily customizable for more classes
- Can be extended with Grad-CAM visualizations
- Perfect as a deployable portfolio project
