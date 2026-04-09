# 🦠 Black Fungus Detection using Deep Learning

## 📌 Overview

This project focuses on detecting **Black Fungus (Mucormycosis)** from medical images using a **Convolutional Neural Network (CNN)**.
It helps in early diagnosis by classifying images as **Infected** or **Not Infected**.

---

## 🚀 Features

* Image classification using CNN
* Data preprocessing and augmentation
* Binary classification (Infected / Not Infected)
* Simple and scalable architecture

---

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy & Pandas
* Flask (optional for deployment)

---

## ⚙️ Project Structure

```
black-fungus-detection/
│
├── train.py            # Model training
├── predict.py          # Prediction script
├── requirements.txt    # Dependencies
├── README.md           # Project documentation
└── .gitignore
```

---

## 🛠️ How It Works

1. Input image is uploaded
2. Image is preprocessed (resized & normalized)
3. CNN extracts features
4. Model predicts output
5. Result: **Infected / Not Infected**

---

## ▶️ How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Train the model

```
python train.py
```

### 3. Run prediction

```
python predict.py
```

---

## 📊 Output

The model predicts whether the given image is:

* ✅ Infected
* ❌ Not Infected

---

## ⚠️ Limitations

* Requires a proper dataset
* Accuracy depends on training data
* Not a replacement for medical diagnosis

---

## 📌 Future Improvements

* Use larger dataset
* Implement advanced models (ResNet, Xception)
* Add web interface for real-time prediction

---

## 👨‍💻 Author

**Gowtham Reddi**


