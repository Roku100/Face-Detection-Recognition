# Installation Guide

## Prerequisites

- Python 3.12
- Webcam (for live recognition)
- Windows/Linux/Mac

## Setup

### 1. Install Python 3.12

Download from [python.org](https://www.python.org/downloads/)

Make sure to check "Add Python to PATH" during installation.

### 2. Clone/Download Project

```bash
cd "Face Detection & Recognition"
```

### 3. Create Virtual Environment

```bash
python -m venv venv312

# Activate (Windows)
venv312\Scripts\activate

# Activate (Linux/Mac)
source venv312/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This might take a few minutes.

## Verify Installation

```bash
python src/main.py --mode stats
```

You should see system statistics.

## First Run

### Register Your Face

```bash
python scripts/register_face.py --name "Your Name" --mode camera
```

- Look at the camera
- Move your head slightly for different angles
- System auto-captures when quality is good
- Need 5 samples total

### Test Recognition

```bash
cd src
python main.py --mode camera
```

You should see your name appear above your face!

## Troubleshooting

### "No module named cv2"

```bash
pip install opencv-python
```

### "Camera not found"

- Close other apps using the camera
- Try a different camera index: `--input 1`

### "Permission denied" on Linux

```bash
sudo usermod -a -G video $USER
```

Then log out and log back in.

### Import errors

Make sure you're in the virtual environment:
```bash
# You should see (venv312) in your terminal
venv312\Scripts\activate
```

## Optional: Download Pre-trained Models

The DNN detector uses pre-trained models. They're downloaded automatically on first run, but you can also download manually:

```bash
python scripts/download_models.py
```

## Next Steps

See [README.md](README.md) for usage examples.
