import cv2
import numpy as np

class FaceDrawer:
    """Utility class for drawing bounding boxes and labels on images."""
    
    def __init__(self, config):
        self.config = config
        self.bbox_thickness = config.get('display.bbox_thickness', 2)
        self.font_scale = config.get('display.font_scale', 0.6)
        self.known_color = tuple(config.get('display.known_color', [0, 255, 0]))
        self.unknown_color = tuple(config.get('display.unknown_color', [0, 0, 255]))
        self.show_confidence = config.get('display.show_confidence', True)
    
    def draw_face(self, image, bbox, name="Unknown", confidence=None, is_known=True):
        """Draw bounding box and label for a detected face."""
        x, y, w, h = bbox
        
        # Choose color based on whether face is known
        color = self.known_color if is_known else self.unknown_color
        
        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), color, self.bbox_thickness)
        
        # Prepare label text
        if self.show_confidence and confidence is not None:
            label = f"{name} ({confidence:.2f})"
        else:
            label = name
        
        # Calculate label size
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 2
        )
        
        # Draw label background
        cv2.rectangle(
            image,
            (x, y - label_h - baseline - 10),
            (x + label_w + 10, y),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            image,
            label,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            (255, 255, 255),
            2
        )
        
        return image
    
    def draw_multiple_faces(self, image, detections):
        """Draw multiple faces on image.
        
        Args:
            image: Input image
            detections: List of tuples (bbox, name, confidence, is_known)
        """
        for detection in detections:
            bbox, name, confidence, is_known = detection
            image = self.draw_face(image, bbox, name, confidence, is_known)
        
        return image
    
    def draw_fps(self, image, fps):
        """Draw FPS counter on image."""
        text = f"FPS: {fps:.1f}"
        cv2.putText(
            image,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        return image
    
    def draw_info_panel(self, image, info_dict):
        """Draw information panel on image."""
        y_offset = 60
        for key, value in info_dict.items():
            text = f"{key}: {value}"
            cv2.putText(
                image,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            y_offset += 25
        
        return image
