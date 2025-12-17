# 🎓 YOLOv5 Object Detection  
## Streamlit Deployment | MLOps Course Project

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-YOLOv5-red)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![MLOps](https://img.shields.io/badge/MLOps-End--to--End-green)

## 👤 Anggota Tim  
**Kelompok 3 – MLOps RA**

1. Ukasyah Muntaha – 122450028  
2. Abit Ahmad Oktarian – 122450042  
3. Alvia Asrinda br Ginting – 122450077  
4. Uliano Wilyam Purba – 122450098 

---

## 📌 Project Overview

Project ini merupakan implementasi **end-to-end Machine Learning Operations (MLOps)** yang berfokus pada proses **deployment model object detection YOLOv5** ke dalam sebuah **web application interaktif menggunakan Streamlit**.

Aplikasi memungkinkan pengguna melakukan inference object detection pada **image dan video**, dengan menampilkan:
- Bounding box  
- Label kelas  
- Confidence score  

Project ini menekankan aspek:
- Model serving  
- Inference pipeline  
- User interaction  
sebagai bagian dari penerapan MLOps secara menyeluruh.

---

## 🎯 Project Objectives
- Mendeploy model YOLOv5 ke aplikasi web
- Menjalankan inference model secara real-time
- Menyediakan antarmuka pengguna yang interaktif
- Memahami alur dasar Machine Learning Operations (MLOps)

---

## 🧠 Model & Technology Stack

| Component | Technology |
|---------|------------|
| Model | YOLOv5 |
| Task | Object Detection |
| Framework | PyTorch |
| Web App | Streamlit |
| Deployment | Local / Cloud-ready |

---

## 🚀 Application Features
- Input berupa **image** atau **video**
- Sumber data:
  - Example image/video
  - Upload file sendiri
- Pilihan device:
  - CPU
  - CUDA (GPU)
- Visualisasi hasil object detection secara langsung

---

## 🔄 MLOps Workflow

1. **Model Loading**  
   Model YOLOv5 dimuat menggunakan PyTorch dengan opsi pemilihan compute device (CPU atau CUDA).

2. **Data Input Selection**  
   Pengguna memilih sumber data berupa:
   - Example image/video yang disediakan aplikasi, atau  
   - Data image/video yang diunggah sendiri.

3. **Preprocessing**
   - Resize dan normalisasi input  
   - Penyesuaian format data sesuai kebutuhan YOLOv5

4. **Model Inference**
   - Model dijalankan pada device yang dipilih  
   - Proses object detection dilakukan oleh YOLOv5

5. **Postprocessing**
   - Non-Maximum Suppression (NMS)  
   - Ekstraksi bounding box, label kelas, dan confidence score

6. **Visualization**
   - Hasil deteksi ditampilkan langsung pada aplikasi Streamlit

---

## 📂 Project Structure

```bash
YOLOV5-STREAMLIT-DEPLOYMENT/
├── data/
│   ├── example_images/
│   ├── example_videos/
│   ├── uploads/
│   ├── outputs/
│   ├── video_frames/
│   └── video_output/
│
├── models/
│   ├── yoloTrained.pt
│   └── yoloTrained (1).pt
│
├── app.py
├── video_predict.py
├── requirements.txt
├── packages.txt
├── README.md
├── LICENSE
├── .gitignore
└── pre-commit-config.yaml
