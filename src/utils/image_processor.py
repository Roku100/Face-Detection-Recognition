import cv2
import numpy as np
from PIL import Image

class ImageProcessor:
    """Utility class for image preprocessing and manipulation."""
    
    @staticmethod
    def load_image(image_path):
        """Load image from file path."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return image
    
    @staticmethod
    def resize_image(image, width=None, height=None, maintain_aspect=True):
        """Resize image to specified dimensions."""
        if width is None and height is None:
            return image
        
        h, w = image.shape[:2]
        
        if maintain_aspect:
            if width is not None:
                ratio = width / w
                height = int(h * ratio)
            else:
                ratio = height / h
                width = int(w * ratio)
        
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def enhance_image(image):
        """Enhance image quality for better face detection."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    @staticmethod
    def assess_quality(face_image):
        """Assess face image quality based on sharpness and brightness."""
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 500.0, 1.0)
        
        # Calculate brightness
        brightness = np.mean(gray) / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2
        
        # Combined quality score
        quality_score = (sharpness_score * 0.6 + brightness_score * 0.4)
        
        return quality_score
    
    @staticmethod
    def crop_face(image, bbox, padding=0.2):
        """Crop face region with optional padding."""
        x, y, w, h = bbox
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        return image[y1:y2, x1:x2]
    
    @staticmethod
    def convert_to_rgb(image):
        """Convert BGR image to RGB."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def convert_to_bgr(image):
        """Convert RGB image to BGR."""
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def normalize_face(face_image, size=(160, 160)):
        """Normalize face image to standard size."""
        return cv2.resize(face_image, size, interpolation=cv2.INTER_AREA)
