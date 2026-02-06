import time
import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from detection.detector_factory import DetectorFactory
from utils.config import get_config

def benchmark_detectors():
    """Benchmark different face detection methods."""
    
    print("=" * 70)
    print("FACE DETECTION BENCHMARK")
    print("=" * 70)
    
    config = get_config("../config.yaml")
    methods = ['haar', 'dnn']
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Capture test frame
    ret, test_frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not capture frame")
        return
    
    print(f"\nTest image size: {test_frame.shape[1]}x{test_frame.shape[0]}")
    print(f"Running 50 iterations per method...\n")
    
    results = {}
    
    for method in methods:
        print(f"Testing {method.upper()}...")
        
        try:
            # Create detector
            config.set('detection.method', method)
            detector = DetectorFactory.create_detector(method, config)
            
            # Warmup
            for _ in range(5):
                detector.detect(test_frame)
            
            # Benchmark
            times = []
            face_counts = []
            
            for i in range(50):
                start = time.time()
                faces = detector.detect(test_frame)
                elapsed = time.time() - start
                
                times.append(elapsed)
                face_counts.append(len(faces))
            
            avg_time = sum(times) / len(times)
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
            avg_faces = sum(face_counts) / len(face_counts)
            
            results[method] = {
                'avg_time': avg_time,
                'avg_fps': avg_fps,
                'avg_faces': avg_faces,
                'min_time': min(times),
                'max_time': max(times)
            }
            
            print(f"  ✓ Average time: {avg_time*1000:.2f}ms")
            print(f"  ✓ Average FPS: {avg_fps:.1f}")
            print(f"  ✓ Faces detected: {avg_faces:.1f}")
            print()
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
    
    # Print comparison
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<15} {'Avg Time':<15} {'FPS':<15} {'Faces':<15}")
    print("-" * 70)
    
    for method, data in results.items():
        print(f"{method.upper():<15} {data['avg_time']*1000:>10.2f}ms {data['avg_fps']:>10.1f} {data['avg_faces']:>10.1f}")
    
    print("=" * 70)
    
    # Recommendation
    print("\nRECOMMENDATIONS:")
    print("- For real-time video: Use HAAR (fastest)")
    print("- For accuracy: Use DNN (best balance)")
    print("- For best quality: Use MTCNN (slowest, requires tensorflow)")

if __name__ == "__main__":
    benchmark_detectors()
