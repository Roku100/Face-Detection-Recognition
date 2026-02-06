import cv2
import numpy as np
from pathlib import Path

class FaceEncoder:
    def __init__(self, config):
        self.config = config
        self.model_path = config.get('paths.models')
        self._load_face_detector()
        self.face_size = (200, 200)
    
    def _load_face_detector(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_detector = cv2.CascadeClassifier(cascade_path)
    
    def encode_face(self, image, face_location=None):
        if face_location is None:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return None
            
            x, y, w, h = faces[0]
            face_location = (y, x + w, y + h, x)
        
        top, right, bottom, left = face_location
        
        top = max(0, top)
        left = max(0, left)
        bottom = min(image.shape[0], bottom)
        right = min(image.shape[1], right)
        
        face_img = image[top:bottom, left:right]
        
        if face_img.size == 0:
            return None
        
        face_img = cv2.resize(face_img, self.face_size)
        
        if len(face_img.shape) == 3:
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_img
        
        gray_face = cv2.equalizeHist(gray_face)
        
        # Use LBP for better features
        lbp = self._compute_lbp(gray_face)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        
        encoding = hist.astype(np.float32)
        encoding = encoding / (np.linalg.norm(encoding) + 1e-7)
        
        return encoding
    
    def _compute_lbp(self, image):
        # Optimized LBP using numpy vectorization for better performance
        height, width = image.shape
        lbp = np.zeros((height - 2, width - 2), dtype=np.uint8)
        
        # Center pixel
        center = image[1:-1, 1:-1]
        
        # Binary comparison for each neighbor
        lbp |= ((image[0:-2, 0:-2] >= center) << 7).astype(np.uint8)
        lbp |= ((image[0:-2, 1:-1] >= center) << 6).astype(np.uint8)
        lbp |= ((image[0:-2, 2:] >= center) << 5).astype(np.uint8)
        lbp |= ((image[1:-1, 2:] >= center) << 4).astype(np.uint8)
        lbp |= ((image[2:, 2:] >= center) << 3).astype(np.uint8)
        lbp |= ((image[2:, 1:-1] >= center) << 2).astype(np.uint8)
        lbp |= ((image[2:, 0:-2] >= center) << 1).astype(np.uint8)
        lbp |= ((image[1:-1, 0:-2] >= center) << 0).astype(np.uint8)
        
        # Padding back to original size
        return np.pad(lbp, 1, mode='constant')
    
    def encode_faces(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        results = []
        for (x, y, w, h) in faces:
            face_location = (y, x + w, y + h, x)
            encoding = self.encode_face(image, face_location)
            if encoding is not None:
                results.append((encoding, face_location))
        
        return results
    
    def encode_from_bbox(self, image, bbox):
        x, y, w, h = bbox
        face_location = (y, x + w, y + h, x)
        return self.encode_face(image, face_location)
    
    @staticmethod
    def bbox_to_face_location(bbox):
        x, y, w, h = bbox
        return (y, x + w, y + h, x)
    
    @staticmethod
    def face_location_to_bbox(face_location):
        top, right, bottom, left = face_location
        return (left, top, right - left, bottom - top)
