import cv2
import numpy as np

class MTCNNDetector:
    """Face detector using MTCNN - highest accuracy, slower."""
    
    def __init__(self, config):
        self.config = config
        self.confidence_threshold = config.get('detection.confidence_threshold', 0.5)
        
        try:
            from mtcnn import MTCNN
            self.detector = MTCNN()
        except ImportError:
            raise ImportError(
                "MTCNN not installed. Install with: pip install mtcnn tensorflow"
            )
    
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
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.detector.detect_faces(rgb_image)
        
        faces = []
        for result in results:
            confidence = result['confidence']
            
            # Filter by confidence
            if confidence > self.confidence_threshold:
                x, y, w, h = result['box']
                
                # Ensure positive dimensions
                x = max(0, x)
                y = max(0, y)
                w = max(0, w)
                h = max(0, h)
                
                faces.append(((x, y, w, h), confidence))
        
        return faces
    
    def get_name(self):
        """Return detector name."""
        return "MTCNN"
