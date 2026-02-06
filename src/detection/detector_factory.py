from .haar_detector import HaarCascadeDetector
from .dnn_detector import DNNDetector
from .mtcnn_detector import MTCNNDetector

class DetectorFactory:
    """Factory class for creating face detectors."""
    
    DETECTORS = {
        'haar': HaarCascadeDetector,
        'dnn': DNNDetector,
        'mtcnn': MTCNNDetector
    }
    
    @staticmethod
    def create_detector(method, config):
        """Create a face detector based on the specified method.
        
        Args:
            method: Detection method ('haar', 'dnn', or 'mtcnn')
            config: Configuration object
            
        Returns:
            Face detector instance
        """
        if method not in DetectorFactory.DETECTORS:
            raise ValueError(
                f"Unknown detection method: {method}. "
                f"Available methods: {list(DetectorFactory.DETECTORS.keys())}"
            )
        
        detector_class = DetectorFactory.DETECTORS[method]
        
        try:
            return detector_class(config)
        except Exception as e:
            raise RuntimeError(f"Failed to create {method} detector: {e}")
    
    @staticmethod
    def get_available_methods():
        """Get list of available detection methods."""
        return list(DetectorFactory.DETECTORS.keys())
