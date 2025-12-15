# **🎓 YOLOv5 Object Detection – Streamlit Deployment**

### MLOps Course Project

Project ini merupakan project MLOps yang bertujuan untuk mengimplementasikan deployment model machine learning ke dalam sebuah web application interaktif menggunakan Streamlit.
Model yang digunakan adalah **YOLOv5** untuk object detection, yang mampu mendeteksi objek pada gambar maupun video dengan menampilkan bounding box, label kelas, dan confidence score.
Project ini menekankan pada aspek model serving, inference pipeline, dan user interaction, sebagai bagian dari penerapan konsep MLOps secara end-to-end.

---

## 📂 Struktur Direktori
Struktur direktori project disusun untuk memisahkan data, model, dan logic aplikasi agar mudah dikelola dan dikembangkan.

```bash
YOLOV5-STREAMLIT-DEPLOYMENT/
├── __pycache__/                 # Cache file Python
├── .venv/                       # Virtual environment
│
├── data/                        # Data input & output
│   ├── example_images/          # Contoh gambar
│   ├── example_videos/          # Contoh video
│   ├── images/                  # Image hasil proses
│   ├── uploads/                 # File yang diupload user
│   ├── outputs/                 # Output hasil deteksi image
│   ├── video_frames/            # Frame hasil ekstraksi video
│   └── video_output/            # Video hasil object detection
│
├── models/                      # Model dan weight
│   ├── yoloTrained.pt           # Model YOLOv5 terlatih
│   └── yoloTrained (1).pt       # Backup model
│
├── app.py                       # Streamlit main application
├── video_predict.py             # Logic inference untuk video
│
├── requirements.txt             # Python dependencies
├── packages.txt                 # Package tambahan (deployment)
├── LICENSE                      # Lisensi project
├── README.md                    # Dokumentasi project
├── .gitignore                   # Git ignore rules
└── pre-commit-config.yaml       # Pre-commit configuration

---

## **Tujuan Project**
- Mendeploy model YOLOv5 ke aplikasi web
- Mengimplementasikan inference model secara real-time
- Menyediakan antarmuka interaktif untuk pengguna
- Memahami alur dasar Machine Learning Operations (MLOps)

---

## **🧠 Model & Teknologi**
**Model**: YOLOv5
**Task**: Object Detection
**Framework**: PyTorch
**Deployment**: Streamlit

---

## **Fitur Aplikasi**
1. Input berupa image atau video
2. Pilihan sumber data:
    - Example data
    - Upload data sendiri
3. Pilihan device:
    - CPU
    - CUDA (GPU)
4. Visualisasi hasil deteksi objek

---

## 🔄 Workflow Project

Workflow project ini mengikuti alur dasar **Machine Learning Operations (MLOps)** sebagai berikut:

1. **Model Loading**  
   Model YOLOv5 dimuat menggunakan framework PyTorch, dengan opsi pemilihan compute device (CPU atau CUDA).

2. **Input Data Selection**  
   Pengguna memilih sumber data berupa:
   - Example image/video yang disediakan aplikasi, atau
   - Data image/video yang diunggah sendiri.

3. **Preprocessing**
   - Resize dan normalisasi input
   - Penyesuaian format data sesuai kebutuhan model YOLOv5

4. **Model Inference**
   - Model YOLOv5 dijalankan pada device yang dipilih (CPU atau CUDA)
   - Model melakukan object detection pada input data

5. **Postprocessing**
   - Non-Maximum Suppression (NMS)
   - Pengambilan bounding box, label kelas, dan confidence score

6. **Visualisasi & Output**
   - Hasil deteksi ditampilkan dalam bentuk bounding box
   - Output divisualisasikan langsung pada aplikasi Streamlit

---

## **👤 Anggota Tim**
Kelompok 3 MLOPS RA
1. Ukasyah Muntaha - 122450028
2. Abit Ahmad Oktarian - 122450042
3. Alvia Asrinda br Ginting - 122450077
4. Uliano Wilyam Purba - 122450098









