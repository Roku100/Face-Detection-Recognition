import cv2
import numpy as np
import os
from pathlib import Path

class DNNDetector:
    """Face detector using DNN (Deep Neural Network) - more accurate than Haar."""
    
    def __init__(self, config):
        self.config = config
        self.confidence_threshold = config.get('detection.confidence_threshold', 0.5)
        self.model_path = config.get('paths.models')
        
        # Download and load pre-trained model
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained DNN model (ResNet-based SSD)."""
        model_dir = Path(self.model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        prototxt_path = model_dir / "deploy.prototxt"
        model_path = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
        
        # If models don't exist, provide download instructions
        if not prototxt_path.exists() or not model_path.exists():
            self._download_models(prototxt_path, model_path)
        
        # Load the model
        self.net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
    
    def _download_models(self, prototxt_path, model_path):
        """Download pre-trained models if not present."""
        import urllib.request
        
        print("Downloading DNN models (this may take a moment)...")
        
        # URLs for the models
        prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        
        try:
            # Download prototxt
            if not prototxt_path.exists():
                print("Downloading deploy.prototxt...")
                urllib.request.urlretrieve(prototxt_url, prototxt_path)
            
            # Download model
            if not model_path.exists():
                print("Downloading model weights...")
                urllib.request.urlretrieve(model_url, model_path)
            
            print("Models downloaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to download models: {e}")
    
    def detect(self, image):
        """Detect faces in image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of bounding boxes [(x, y, w, h), ...]
        """
        detections = self.detect_with_confidence(image)
        return [bbox for bbox, _ in detections]
    
    def detect_with_confidence(self, image):
        """Detect faces with confidence scores.
        
        Returns:
            List of tuples [(bbox, confidence), ...]
        """
        h, w = image.shape[:2]
        
        # Prepare blob from image
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )
        
        # Pass blob through network
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        
        # Process detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter by confidence
            if confidence > self.confidence_threshold:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                
                # Convert to (x, y, w, h) format
                x = max(0, x1)
                y = max(0, y1)
                width = min(w - x, x2 - x1)
                height = min(h - y, y2 - y1)
                
                faces.append(((x, y, width, height), float(confidence)))
        
        return faces
    
    def get_name(self):
        """Return detector name."""
        return "DNN (ResNet-SSD)"
