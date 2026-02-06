import yaml
import os
from pathlib import Path

class Config:
    """Configuration manager for the face detection and recognition system."""
    
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._ensure_directories()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.get('paths.known_faces'),
            self.get('paths.encodings'),
            self.get('paths.models'),
            self.get('paths.logs')
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get(self, key, default=None):
        """Get configuration value using dot notation (e.g., 'detection.method')."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default
    
    def set(self, key, value):
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self):
        """Save current configuration to file."""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

# Global config instance
_config = None

def get_config(config_path="config.yaml"):
    """Get or create global configuration instance."""
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config
