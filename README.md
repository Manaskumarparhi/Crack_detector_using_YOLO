# UAV-Based Crack Detection System

A deep learning system to detect cracks and structural defects in roads, buildings, and infrastructure using drone (UAV) imagery. Built with YOLOv8 and deployed as an interactive web interface using Gradio.

---

## Demo

Upload an image, video, or connect a live drone stream to detect cracks and get real-world length measurements in mm, cm, and metres.


---

## Project Structure

```
crack-detection/
├── UAV_Crack_Detection_Training.ipynb   # Full training pipeline (run in Google Colab)
├── app.py                               # Gradio web interface for inference
├── requirements.txt                     # Python dependencies
└── README.md
```

> `best.pt` (trained model weights) is not included in this repository.  
> Run the Colab notebook to generate it, then place it in this folder.

---

## Dataset

**UAV-Based Crack Detection Dataset** by ziya07 on Kaggle  
Link: https://www.kaggle.com/datasets/ziya07/uav-based-crack-detection-dataset

- 630 PNG images of cracked surfaces captured from drones
- Paired binary segmentation masks (white = crack, black = background)
- Covers roads, concrete, and construction surfaces

---

## Training (Google Colab)

### Step 1 — Open the notebook in Colab

[![Open In Colab][(https://colab.research.google.com/)]

### Step 2 — Set runtime to GPU 

In Colab: `Runtime → Change runtime type → T4 GPU`

### Step 3 — Run all cells in order

The notebook will:
1. Install all dependencies
2. Download the dataset from Kaggle automatically
3. Convert binary masks to YOLO bounding box labels
4. Split data into 80% train / 10% val / 10% test
5. Train a YOLOv8s model for 50 epochs
6. Evaluate on the test set and plot training curves
7. Download `best.pt` to your machine

### Training Results

| Metric | Score |
|--------|-------|
| mAP50 | 0.394 |
| mAP50-95 | 0.208 |
| Precision | 0.456 |
| Recall | 0.533 |

---

## Running the Web Interface (VS Code / Local)

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/crack-detection.git
cd crack-detection
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Add your trained weights

Place the `best.pt` file (downloaded from Colab) into the project folder:

```
crack-detection/
├── best.pt          ← put it here
├── app.py
└── ...
```

### Step 4 — Run the app

```bash
python app.py
```

Then open your browser at: **http://127.0.0.1:7860**

---

## Interface Features

The Gradio web app has 5 tabs:

| Tab | Input | Description |
|-----|-------|-------------|
| Single image | Upload `.jpg` / `.png` | Detects cracks and shows full measurements |
| Folder of images | Upload multiple images | Batch processing + downloadable CSV report |
| Video file | Upload `.mp4` / `.avi` | Annotated output video with crack overlays |
| Live webcam | Browser webcam | Real-time detection from webcam |
| Stream URL | RTSP / HTTP URL | Live drone feed over Wi-Fi |

---

## Crack Length Measurement

Each detected crack is measured using **Ground Sampling Distance (GSD)**:

```
GSD (mm/px) = (altitude_m × 1000 × sensor_width_mm) / (focal_length_mm × image_width_px)
crack_length = bbox_diagonal_px × GSD
```

For each crack the app reports:

- Bounding box coordinates (pixels)
- Width, height, diagonal (pixels)
- Length in mm, cm, and metres

Configure your drone's altitude, focal length, and sensor width in the app's settings panel for accurate real-world measurements.

### Default drone settings

| Parameter | Default | Notes |
|-----------|---------|-------|
| Altitude | 10 m | Change to your actual flight height |
| Focal length | 4.5 mm | Typical DJI Mini / Phantom camera |
| Sensor width | 6.17 mm | 1/2.3" sensor |

---

## Model Details

| Property | Value |
|----------|-------|
| Architecture | YOLOv8s |
| Task | Object detection |
| Input size | 640 × 640 |
| Classes | 1 (crack) |
| Parameters | 11.1M |
| Training epochs | 50 |
| Framework | Ultralytics |

---

## Tech Stack

- **Model:** YOLOv8 (Ultralytics)
- **Training:** Google Colab (Tesla T4 GPU)
- **Interface:** Gradio
- **Image processing:** OpenCV, NumPy
- **Dataset:** Kaggle (kagglehub)
- **Visualization:** Matplotlib

---

## Use Cases

- Road surface inspection from drone footage
- Bridge and infrastructure crack monitoring
- Construction site quality control
- Automated defect reporting with real-world measurements

---

## Author

**Manas Kumar Parhi**  
B.Tech Computer Science — IGIT Sarang  
BS Data Science & AI — IIT Madras  

- GitHub: [github.com/Manaskumarparhi](https://github.com/Manaskumarparhi)  
- LinkedIn: [linkedin.com/in/manas-kumar-parhi](https://www.linkedin.com/in/manas-kumar-parhi)  
- Email: manaskumarparhi73@gmail.com

---

## License

This project is open source and available under the [MIT License](LICENSE).
