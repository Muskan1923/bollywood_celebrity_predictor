# bollywood_celebrity_predictor
# 🎬 Bollywood Celebrity Predictor  

A *Deep Learning and Computer Vision project* that predicts *which Bollywood celebrity you resemble the most*.  
This project combines *face detection, **feature extraction, and **similarity matching* techniques to deliver accurate and fun results.  

Built with *TensorFlow/Keras, **VGGFace (ResNet50), **MTCNN, **OpenCV, and a **Streamlit UI*, this project demonstrates how pre-trained models can be applied to solve real-world face recognition tasks.  

---

## ✨ Project Overview  

Have you ever wondered which Bollywood celebrity you look like?  
This project allows you to upload your picture, detects your face, extracts facial embeddings using *VGGFace*, and then compares them with embeddings of Bollywood celebrity images stored in the dataset.  

The closest match is returned along with the celebrity's image.  

🔑 *Key Concepts Used*:  
- *Face Detection: Using **MTCNN* to locate the face in an image.  
- *Feature Extraction: Using **VGGFace (ResNet50)* without the top layer, trained on large face datasets.  
- *Cosine Similarity*: Measuring similarity between extracted embeddings and stored celebrity embeddings.  
- *Interactive App: Built with **Streamlit* for a user-friendly interface.  

---

## 📂 Project Structure  

The repository is organized as follows:
---

## 🛠 Tech Stack  

- *Programming Language*: Python  
- *Deep Learning Framework*: TensorFlow / Keras  
- *Face Recognition Model*: VGGFace (ResNet50 backbone)  
- *Face Detection*: MTCNN (Multi-task Cascaded Convolutional Neural Networks)  
- *Image Processing*: OpenCV & Pillow  
- *Similarity Measure*: Cosine Similarity (scikit-learn)  
- *Web Framework*: Streamlit  

---


### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/muskan1923/bollywood-celebrity-predictor.git
cd bollywood-celebrity-predictor
