import cv2
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from recognition.face_encoder import FaceEncoder
from recognition.database_manager import FaceDatabase
from utils.config import get_config
from utils.image_processor import ImageProcessor

class FaceRegistration:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = str(Path(__file__).parent.parent / "config.yaml")
        
        self.config = get_config(config_path)
        self.encoder = FaceEncoder(self.config)
        self.database = FaceDatabase(self.config)
        self.image_processor = ImageProcessor()
        
        # Get settings from config
        self.min_samples = self.config.get('registration.min_samples', 5)
        self.quality_threshold = self.config.get('registration.quality_threshold', 0.7)
        self.auto_capture_interval = self.config.get('registration.auto_capture_interval', 0.5)
    
    def register_from_images(self, name, image_paths):
        """Register a person from a list of images.
        
        Args:
            name: Person's name
            image_paths: List of image file paths
        """
        print(f"\n=== Registering: {name} ===")
        print(f"Processing {len(image_paths)} image(s)...")
        
        encodings = []
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                # Load image
                image = self.image_processor.load_image(image_path)
                
                # Generate encoding
                encoding = self.encoder.encode_face(image)
                
                if encoding is not None:
                    # Assess quality
                    quality = self.image_processor.assess_quality(image)
                    
                    if quality >= self.quality_threshold:
                        encodings.append(encoding)
                        print(f"  [OK] Image {i}/{len(image_paths)}: Quality {quality:.2f}")
                    else:
                        print(f"  [LOW] Image {i}/{len(image_paths)}: Low quality {quality:.2f}")
                else:
                    print(f"  [X] Image {i}/{len(image_paths)}: No face detected")
            
            except Exception as e:
                print(f"  [ERR] Image {i}/{len(image_paths)}: Error - {e}")
        
        # Save to database
        if len(encodings) >= self.min_samples:
            for encoding in encodings:
                self.database.add_person(name, encoding)
            
            self.database.save_database()
            print(f"\n[SUCCESS] Successfully registered {name} with {len(encodings)} encoding(s)")
            return True
        else:
            print(f"\n[FAILED] Failed: Need at least {self.min_samples} samples")
            print(f"  Only got {len(encodings)} valid encoding(s)")
            return False
    
    def register_from_camera(self, name, camera_id=0):
        """Register a person using camera with auto-capture.
        
        Args:
            name: Person's name
            camera_id: Camera device ID
        """
        print(f"\n=== Registering: {name} ===")
        print(f"Opening camera {camera_id}...")
        print("\nInstructions:")
        print("  - Look at the camera from different angles")
        print("  - System will auto-capture when face quality is good")
        print(f"  - Need {self.min_samples} good samples")
        print("  - Press 'q' to quit, 'c' to force capture")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"✗ Could not open camera {camera_id}")
            return False
        
        encodings = []
        last_capture_time = 0
        
        try:
            while len(encodings) < self.min_samples:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Create display frame
                display_frame = frame.copy()
                
                # Try to encode face
                encoding = self.encoder.encode_face(frame)
                
                if encoding is not None:
                    # Assess quality
                    quality = self.image_processor.assess_quality(frame)
                    
                    # Draw quality indicator
                    color = (0, 255, 0) if quality >= self.quality_threshold else (0, 165, 255)
                    cv2.putText(
                        display_frame,
                        f"Quality: {quality:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )
                    
                    # Auto-capture if quality is good and enough time has passed
                    current_time = time.time()
                    if (quality >= self.quality_threshold and 
                        current_time - last_capture_time >= self.auto_capture_interval):
                        
                        encodings.append(encoding)
                        last_capture_time = current_time
                        
                        # Visual feedback
                        cv2.rectangle(display_frame, (0, 0), 
                                    (display_frame.shape[1], display_frame.shape[0]),
                                    (0, 255, 0), 10)
                        print(f"  ✓ Captured {len(encodings)}/{self.min_samples}")
                else:
                    cv2.putText(
                        display_frame,
                        "No face detected",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
                
                # Draw progress
                cv2.putText(
                    display_frame,
                    f"Samples: {len(encodings)}/{self.min_samples}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                cv2.imshow('Face Registration', display_frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[CANCELLED] Registration cancelled")
                    return False
                elif key == ord('c') and encoding is not None:
                    encodings.append(encoding)
                    print(f"  [AUTO] Manual capture {len(encodings)}/{self.min_samples}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Save to database
        if len(encodings) >= self.min_samples:
            for encoding in encodings:
                self.database.add_person(name, encoding)
            
            self.database.save_database()
            print(f"\n[SUCCESS] Successfully registered {name} with {len(encodings)} encoding(s)")
            return True
        else:
            print(f"\n[FAILED] Failed: Need at least {self.min_samples} samples")
            return False
    
    def list_registered_people(self):
        """List all registered people."""
        names = self.database.get_person_names()
        
        if not names:
            print("\nNo people registered yet")
            return
        
        print(f"\n=== Registered People ({len(names)}) ===")
        for i, name in enumerate(sorted(names), 1):
            person_data = self.database.get_person(name)
            num_encodings = len(person_data['encodings'])
            added_date = person_data['metadata'].get('added_date', 'Unknown')
            print(f"{i}. {name} - {num_encodings} encoding(s) - Added: {added_date}")
    
    def remove_person(self, name):
        """Remove a person from database."""
        if self.database.remove_person(name):
            self.database.save_database()
            print(f"[SUCCESS] Removed {name} from database")
            return True
        else:
            print(f"[FAILED] {name} not found in database")
            return False

def main():
    """Main entry point for registration script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Registration System')
    parser.add_argument('--name', required=True, help='Name of person to register')
    parser.add_argument('--mode', choices=['camera', 'images'], default='camera',
                       help='Registration mode')
    parser.add_argument('--images', nargs='+', help='Image paths (for images mode)')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--list', action='store_true', help='List registered people')
    parser.add_argument('--remove', help='Remove person from database')
    
    args = parser.parse_args()
    
    # Initialize registration system
    registration = FaceRegistration()
    
    # Handle list command
    if args.list:
        registration.list_registered_people()
        return
    
    # Handle remove command
    if args.remove:
        registration.remove_person(args.remove)
        return
    
    # Register new person
    if args.mode == 'camera':
        registration.register_from_camera(args.name, args.camera)
    elif args.mode == 'images':
        if not args.images:
            print("Error: --images required for images mode")
            return
        registration.register_from_images(args.name, args.images)

if __name__ == "__main__":
    main()
