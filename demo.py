"""
Quick Start Demo Script
This script demonstrates the basic functionality of the face recognition system.
"""

import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from detection.detector_factory import DetectorFactory
from utils.config import get_config
from utils.drawer import FaceDrawer

def test_detection():
    """Test face detection on webcam."""
    print("=" * 60)
    print("FACE DETECTION & RECOGNITION SYSTEM - QUICK DEMO")
    print("=" * 60)
    
    # Load configuration
    config = get_config("config.yaml")
    
    # Test different detection methods
    methods = ['haar', 'dnn']
    
    for method in methods:
        print(f"\n▶ Testing {method.upper()} detector...")
        
        try:
            # Create detector
            config.set('detection.method', method)
            detector = DetectorFactory.create_detector(method, config)
            drawer = FaceDrawer(config)
            
            print(f"✓ {detector.get_name()} loaded successfully")
            
            # Open camera
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("✗ Could not open camera")
                continue
            
            print("  Press 'q' to test next detector, 's' to skip to next")
            
            frame_count = 0
            while frame_count < 100:  # Test for 100 frames
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect faces every 5 frames
                if frame_count % 5 == 0:
                    detections = detector.detect_with_confidence(frame)
                    
                    # Draw detections
                    for bbox, confidence in detections:
                        x, y, w, h = bbox
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{confidence:.2f}", (x, y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Add info
                cv2.putText(frame, f"Detector: {method.upper()}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to continue", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Detection Test', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('s'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            print(f"✓ {method.upper()} test completed")
            
        except Exception as e:
            print(f"✗ Error testing {method}: {e}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Register faces: python scripts/register_face.py --name 'Your Name' --mode camera")
    print("2. Run recognition: cd src && python main.py --mode camera")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    test_detection()
