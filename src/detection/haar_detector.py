import cv2
import os

class HaarCascadeDetector:
    """Face detector using Haar Cascades - fast but less accurate."""
    
    def __init__(self, config):
        self.config = config
        self.scale_factor = config.get('detection.scale_factor', 1.1)
        self.min_neighbors = config.get('detection.min_neighbors', 5)
        self.min_size = tuple(config.get('detection.min_face_size', [30, 30]))
        
        # Load Haar Cascade classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise ValueError("Failed to load Haar Cascade classifier")
    
    def detect(self, image):
        """Detect faces in image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of bounding boxes [(x, y, w, h), ...]
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert to list of tuples
        return [tuple(face) for face in faces]
    
    def detect_with_confidence(self, image):
        """Detect faces with confidence scores.
        
        Note: Haar Cascades don't provide confidence scores,
        so we return 1.0 for all detections.
        
        Returns:
            List of tuples [(bbox, confidence), ...]
        """
        faces = self.detect(image)
        return [(face, 1.0) for face in faces]
    
    def get_name(self):
        """Return detector name."""
        return "Haar Cascade"
