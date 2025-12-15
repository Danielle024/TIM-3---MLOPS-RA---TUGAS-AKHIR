import os
import glob
from datetime import datetime

import streamlit as st
import torch
from PIL import Image
import wget

from video_predict import runVideo


# =========================
# Configurations
# =========================
CFG_MODEL_PATH = "models/yourModel.pt"
CFG_ENABLE_URL_DOWNLOAD = True
CFG_ENABLE_VIDEO_PREDICTION = True

# If CFG_ENABLE_URL_DOWNLOAD = True:
URL_MODEL = "https://archive.org/download/yoloTrained/yoloTrained.pt"
# =========================


# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/outputs", exist_ok=True)


def _safe_filename(name: str) -> str:
    return os.path.basename(name).replace(" ", "_")


@st.cache_resource
def downloadModel():
    """Download model weights once (if enabled)."""
    if not CFG_ENABLE_URL_DOWNLOAD:
        return

    filename = URL_MODEL.split("/")[-1]
    outpath = os.path.join("models", filename)

    if not os.path.exists(outpath):
        wget.download(URL_MODEL, out="models/")


@st.cache_resource
def loadmodel(device: str):
    """Load YOLOv5 model once per device."""
    if CFG_ENABLE_URL_DOWNLOAD:
        model_path = os.path.join("models", URL_MODEL.split("/")[-1])
    else:
        model_path = CFG_MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=model_path,
        device=device
    )
    return model


def imageInput(model, src):
    if src == "Upload your own data.":
        image_file = st.file_uploader("Upload An Image", type=["png", "jpeg", "jpg"])
        if image_file is None:
            return

        col1, col2 = st.columns(2)

        # ✅ FIX: tampilkan gambar sebagai bytes agar Streamlit gak re-encode via PIL JPEG encoder
        with col1:
            st.image(
                image_file.getvalue(),
                caption="Uploaded Image",
                use_container_width=True
            )

        # Tetap bikin versi PIL untuk kebutuhan lain (kalau perlu)
        _ = Image.open(image_file).convert("RGB")

        ts = int(datetime.timestamp(datetime.now()))
        fname = _safe_filename(image_file.name)
        imgpath = os.path.join("data/uploads", f"{ts}_{fname}")
        outputpath = os.path.join("data/outputs", f"{ts}_{fname}")

        # Simpan file untuk inferensi YOLO
        with open(imgpath, mode="wb") as f:
            f.write(image_file.getbuffer())

        with st.spinner("Predicting..."):
            pred = model(imgpath)
            pred.render()

            # pred.ims = list of numpy arrays
            if pred.ims and len(pred.ims) > 0:
                Image.fromarray(pred.ims[0]).save(outputpath)

        # ✅ FIX: tampilkan output sebagai PNG untuk menghindari jalur encoder JPEG
        with col2:
            if os.path.exists(outputpath):
                st.image(
                    Image.open(outputpath).convert("RGB"),
                    caption="Model Prediction(s)",
                    use_container_width=True,
                    output_format="PNG"
                )
            else:
                st.error("Prediction output not found. Check model inference step.")

    elif src == "From example data.":
        imgpaths = glob.glob("data/example_images/*")
        if len(imgpaths) == 0:
            st.error("No images found. Put example images in data/example_images.", icon="⚠️")
            return

        imgsel = st.slider(
            "Select random images from example data.",
            min_value=1,
            max_value=len(imgpaths),
            step=1,
        )

        image_path = imgpaths[imgsel - 1]
        submit = st.button("Predict!")
        col1, col2 = st.columns(2)

        with col1:
            # ✅ FIX: output_format PNG
            st.image(
                Image.open(image_path).convert("RGB"),
                caption="Selected Image",
                use_container_width=True,
                output_format="PNG"
            )

        if not submit:
            return

        outname = _safe_filename(os.path.basename(image_path))
        outpath = os.path.join("data/outputs", outname)

        with st.spinner("Predicting..."):
            pred = model(image_path)
            pred.render()
            if pred.ims and len(pred.ims) > 0:
                Image.fromarray(pred.ims[0]).save(outpath)

        with col2:
            if os.path.exists(outpath):
                st.image(
                    Image.open(outpath).convert("RGB"),
                    caption="Model Prediction(s)",
                    use_container_width=True,
                    output_format="PNG"
                )
            else:
                st.error("Prediction output not found. Check model inference step.")


def videoInput(model, src):
    pred_view = st.empty()
    warning = st.empty()  # always defined

    if src == "Upload your own data.":
        uploaded_video = st.file_uploader("Upload A Video", type=["mp4", "mpeg", "mov"])
        if uploaded_video is None:
            return

        ts = int(datetime.timestamp(datetime.now()))
        vname = _safe_filename(uploaded_video.name)
        uploaded_video_path = os.path.join("data/uploads", f"{ts}_{vname}")

        with open(uploaded_video_path, mode="wb") as f:
            f.write(uploaded_video.read())

        with open(uploaded_video_path, "rb") as f:
            st.video(f.read())

        st.write("Uploaded Video")

        if st.button("Run Prediction"):
            runVideo(model, uploaded_video_path, pred_view, warning)

    elif src == "From example data.":
        videopaths = glob.glob("data/example_videos/*")
        if len(videopaths) == 0:
            st.error("No videos found. Put example videos in data/example_videos.", icon="⚠️")
            return

        vsel = st.slider(
            "Select random video from example data.",
            min_value=1,
            max_value=len(videopaths),
            step=1,
        )
        video = videopaths[vsel - 1]

        if st.button("Predict!"):
            runVideo(model, video, pred_view, warning)


def main():
    st.set_page_config(page_title="YOLOv5 Streamlit", layout="wide")

    if CFG_ENABLE_URL_DOWNLOAD:
        downloadModel()
    else:
        if not os.path.exists(CFG_MODEL_PATH):
            st.error("Model not found. Set CFG_ENABLE_URL_DOWNLOAD=True or put weights in models/.", icon="⚠️")
            return

    st.sidebar.title("⚙️ Options")
    datasrc = st.sidebar.radio("Select input source.", ["From example data.", "Upload your own data."])

    if CFG_ENABLE_VIDEO_PREDICTION:
        option = st.sidebar.radio("Select input type.", ["Image", "Video"])
    else:
        option = st.sidebar.radio("Select input type.", ["Image"])

    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("Select compute Device.", ["cpu", "cuda"], index=1)
    else:
        deviceoption = st.sidebar.radio("Select compute Device.", ["cpu", "cuda"], index=0, disabled=True)

    st.header("📦 YOLOv5 Streamlit Deployment Example")

    model = loadmodel(deviceoption)

    if option == "Image":
        imageInput(model, datasrc)
    else:
        videoInput(model, datasrc)


if __name__ == "__main__":
    main()
