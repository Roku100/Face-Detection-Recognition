# Face Detection & Recognition System

A face detection and recognition system built with OpenCV for the Syntexhub internship project.

## Features

- Multiple detection methods (Haar Cascades, DNN)
- Face recognition using Local Binary Patterns
- Real-time video processing
- Interactive face registration
- Support for multiple faces
- Temporal smoothing to reduce flickering

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Register a Face

```bash
python scripts/register_face.py --name "Your Name" --mode camera
```

Follow the on-screen instructions. The system will auto-capture 5 samples when quality is good.

### Run Recognition

```bash
cd src
python main.py --mode camera
```

Press 'q' to quit, 's' to save screenshot.

## Usage

### Process an Image

```bash
python main.py --mode image --input photo.jpg --output result.jpg
```

### Process a Video

```bash
python main.py --mode video --input video.mp4 --output processed.mp4
```

### List Registered People

```bash
python scripts/register_face.py --list
```

### Remove Someone

```bash
python scripts/register_face.py --remove "Name"
```

## Configuration

Edit `config.yaml` to customize settings:

- Detection method (haar, dnn)
- Recognition tolerance
- Video frame skip
- Display options

## Project Structure

```
├── src/
│   ├── detection/          # Face detection modules
│   ├── recognition/        # Face recognition modules
│   ├── utils/              # Utilities
│   └── main.py             # Main application
├── scripts/
│   ├── register_face.py    # Face registration
│   └── benchmark.py        # Performance testing
├── data/
│   ├── encodings/          # Face database
│   ├── known_faces/        # Face images
│   └── models/             # Pre-trained models
└── config.yaml             # Configuration

```

## How It Works

1. **Detection**: Uses OpenCV's Haar Cascade or DNN to detect faces
2. **Encoding**: Extracts Local Binary Pattern features from detected faces
3. **Matching**: Compares new faces against registered database using chi-square distance
4. **Tracking**: Maintains face identity across frames to reduce flickering

## Requirements

- Python 3.12
- OpenCV
- NumPy
- See `requirements.txt` for full list

## Tips

- Good lighting improves accuracy
- Register faces from multiple angles
- Adjust tolerance in config for stricter/looser matching
- Use DNN detection for better accuracy

## Troubleshooting

**Camera not opening**: Close other apps using the camera

**Low FPS**: Switch to Haar Cascade detection or increase frame_skip

**False recognitions**: Lower the tolerance value in config.yaml

**No face detected**: Ensure good lighting and face the camera directly

## License

MIT
