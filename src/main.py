import cv2
import time
from pathlib import Path
import sys
import numpy as np
from collections import deque

sys.path.append(str(Path(__file__).parent))

from detection.detector_factory import DetectorFactory
from recognition.face_encoder import FaceEncoder
from recognition.database_manager import FaceDatabase
from recognition.face_matcher import FaceMatcher
from utils.config import get_config
from utils.drawer import FaceDrawer
from utils.image_processor import ImageProcessor

class FaceTracker:
    def __init__(self, history_size=5, iou_threshold=0.5):
        self.tracks = {}
        self.next_id = 0
        self.history_size = history_size
        self.iou_threshold = iou_threshold
        self.max_age = 10
    
    def _calculate_iou(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, detections):
        # Age existing tracks
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['age'] += 1
            if self.tracks[track_id]['age'] > self.max_age:
                del self.tracks[track_id]
        
        matched_tracks = set()
        results = []
        
        for bbox, name, confidence, is_known in detections:
            best_iou = 0
            best_track_id = None
            
            # Find matching track based on position overlap
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                iou = self._calculate_iou(bbox, track['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                track = self.tracks[best_track_id]
                track['bbox'] = bbox
                track['age'] = 0
                matched_tracks.add(best_track_id)
                
                if name and is_known:
                    if name not in track['name_votes']:
                        track['name_votes'][name] = deque(maxlen=self.history_size)
                    track['name_votes'][name].append(confidence)
                
                if track['name_votes']:
                    best_name = max(track['name_votes'].items(), 
                                  key=lambda x: (len(x[1]), np.mean(x[1])))
                    stable_name = best_name[0]
                    stable_confidence = np.mean(best_name[1])
                else:
                    stable_name = "Unknown"
                    stable_confidence = confidence
                
                results.append((bbox, stable_name, stable_confidence, stable_name != "Unknown"))
            else:
                track_id = self.next_id
                self.next_id += 1
                
                self.tracks[track_id] = {
                    'bbox': bbox,
                    'name_votes': {name: deque([confidence], maxlen=self.history_size)} if name and is_known else {},
                    'age': 0
                }
                matched_tracks.add(track_id)
                
                results.append((bbox, name, confidence, is_known))
        
        return results

class FaceRecognitionSystem:
    def __init__(self, config_path="config.yaml"):
        self.config = get_config(config_path)
        
        detection_method = self.config.get('detection.method', 'dnn')
        self.detector = DetectorFactory.create_detector(detection_method, self.config)
        self.encoder = FaceEncoder(self.config)
        self.database = FaceDatabase(self.config)
        self.matcher = FaceMatcher(self.config, self.database)
        self.drawer = FaceDrawer(self.config)
        self.image_processor = ImageProcessor()
        self.tracker = FaceTracker(history_size=5, iou_threshold=0.5)
        
        print(f"✓ Face Recognition System initialized")
        print(f"✓ Using {self.detector.get_name()} for detection")
        print(f"✓ Database: {len(self.database.get_person_names())} people registered")
    
    def process_image(self, image_path, output_path=None, show=True):
        image = self.image_processor.load_image(image_path)
        detections = self.detector.detect_with_confidence(image)
        
        print(f"\nProcessing: {image_path}")
        print(f"Detected {len(detections)} face(s)")
        
        results = []
        for bbox, det_confidence in detections:
            encoding = self.encoder.encode_from_bbox(image, bbox)
            
            if encoding is not None:
                name, rec_confidence = self.matcher.match_face(encoding)
                
                if name:
                    print(f"  ✓ Recognized: {name} (confidence: {rec_confidence:.2f})")
                    results.append((bbox, name, rec_confidence, True))
                else:
                    print(f"  ? Unknown face")
                    results.append((bbox, "Unknown", det_confidence, False))
            else:
                print(f"  ✗ Could not encode face")
                results.append((bbox, "Unknown", det_confidence, False))
        
        output_image = self.drawer.draw_multiple_faces(image, results)
        
        if output_path:
            cv2.imwrite(output_path, output_image)
            print(f"✓ Saved result to: {output_path}")
        
        if show:
            cv2.imshow('Face Recognition', output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return results
    
    def process_video(self, video_source=0, output_path=None):
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_skip = self.config.get('video.frame_skip', 2)
        show_fps = self.config.get('video.display_fps', True)
        
        frame_count = 0
        fps_start_time = time.time()
        fps_counter = 0
        current_fps = 0
        
        print("\n▶ Starting video processing...")
        print("Press 'q' to quit, 's' to save current frame")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if frame_count % frame_skip == 0:
                    detections = self.detector.detect_with_confidence(frame)
                    
                    results = []
                    for bbox, det_confidence in detections:
                        encoding = self.encoder.encode_from_bbox(frame, bbox)
                        
                        if encoding is not None:
                            name, rec_confidence = self.matcher.match_face(encoding)
                            
                            if name:
                                results.append((bbox, name, rec_confidence, True))
                            else:
                                results.append((bbox, "Unknown", det_confidence, False))
                        else:
                            results.append((bbox, "Unknown", det_confidence, False))
                    
                    results = self.tracker.update(results)
                    frame = self.drawer.draw_multiple_faces(frame, results)
                
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()
                
                if show_fps:
                    frame = self.drawer.draw_fps(frame, current_fps)
                
                if writer:
                    writer.write(frame)
                
                cv2.imshow('Face Recognition - Live', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    save_path = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(save_path, frame)
                    print(f"✓ Saved screenshot: {save_path}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            print("\n✓ Video processing stopped")
    
    def get_stats(self):
        db_stats = self.database.get_stats()
        
        return {
            'detector': self.detector.get_name(),
            'registered_people': db_stats['total_people'],
            'total_encodings': db_stats['total_encodings'],
            'avg_encodings_per_person': db_stats['avg_encodings_per_person']
        }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Detection & Recognition System')
    parser.add_argument('--mode', choices=['image', 'video', 'camera', 'stats'], 
                       default='camera', help='Processing mode')
    parser.add_argument('--input', help='Input image or video path')
    parser.add_argument('--output', help='Output path for results')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--no-display', action='store_true', help='Do not display results')
    
    args = parser.parse_args()
    
    system = FaceRecognitionSystem(args.config)
    
    if args.mode == 'image':
        if not args.input:
            print("Error: --input required for image mode")
            return
        system.process_image(args.input, args.output, not args.no_display)
    
    elif args.mode == 'video':
        if not args.input:
            print("Error: --input required for video mode")
            return
        system.process_video(args.input, args.output)
    
    elif args.mode == 'camera':
        camera_id = int(args.input) if args.input else 0
        system.process_video(camera_id, args.output)
    
    elif args.mode == 'stats':
        stats = system.get_stats()
        print("\n=== System Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
