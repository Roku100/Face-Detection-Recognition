import pickle
import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np

class FaceDatabase:
    """Manage known face encodings and metadata."""
    
    def __init__(self, config):
        self.config = config
        self.db_path = Path(config.get('database.path', 'data/encodings/face_database.pkl'))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Database structure: {name: {'encodings': [], 'metadata': {}}}
        self.database = self._load_database()
    
    def _load_database(self):
        """Load database from file."""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load database: {e}")
                return {}
        return {}
    
    def save_database(self):
        """Save database to file."""
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.database, f)
            
            # Also save JSON metadata for easy inspection
            self._save_metadata()
            
            # Backup if enabled
            if self.config.get('database.backup_enabled', True):
                self._create_backup()
            
            return True
        except Exception as e:
            print(f"Error saving database: {e}")
            return False
    
    def _save_metadata(self):
        """Save human-readable metadata."""
        metadata_path = self.db_path.with_suffix('.json')
        metadata = {}
        
        for name, data in self.database.items():
            metadata[name] = {
                'num_encodings': len(data['encodings']),
                'added_date': data['metadata'].get('added_date', 'Unknown'),
                'last_updated': data['metadata'].get('last_updated', 'Unknown')
            }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _create_backup(self):
        """Create timestamped backup of database."""
        backup_dir = self.db_path.parent / 'backups'
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = backup_dir / f"face_database_{timestamp}.pkl"
        
        try:
            with open(backup_path, 'wb') as f:
                pickle.dump(self.database, f)
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")
    
    def add_person(self, name, encoding, metadata=None):
        """Add a new person or update existing person's encodings.
        
        Args:
            name: Person's name
            encoding: Face encoding (128-D vector)
            metadata: Optional metadata dictionary
        """
        if name not in self.database:
            self.database[name] = {
                'encodings': [],
                'metadata': metadata or {}
            }
            self.database[name]['metadata']['added_date'] = datetime.now().isoformat()
        
        self.database[name]['encodings'].append(encoding)
        self.database[name]['metadata']['last_updated'] = datetime.now().isoformat()
        
        return True
    
    def remove_person(self, name):
        """Remove a person from database."""
        if name in self.database:
            del self.database[name]
            return True
        return False
    
    def get_person(self, name):
        """Get person's data from database."""
        return self.database.get(name)
    
    def get_all_encodings(self):
        """Get all encodings with their names.
        
        Returns:
            Tuple of (encodings_list, names_list)
        """
        encodings = []
        names = []
        
        for name, data in self.database.items():
            for encoding in data['encodings']:
                encodings.append(encoding)
                names.append(name)
        
        return encodings, names
    
    def get_person_names(self):
        """Get list of all registered person names."""
        return list(self.database.keys())
    
    def get_stats(self):
        """Get database statistics."""
        total_people = len(self.database)
        total_encodings = sum(len(data['encodings']) for data in self.database.values())
        
        return {
            'total_people': total_people,
            'total_encodings': total_encodings,
            'avg_encodings_per_person': total_encodings / total_people if total_people > 0 else 0
        }
    
    def clear_database(self):
        """Clear all data from database."""
        self.database = {}
        return self.save_database()
    
    def export_to_json(self, output_path):
        """Export database to JSON format (without encodings)."""
        export_data = {}
        
        for name, data in self.database.items():
            export_data[name] = {
                'num_encodings': len(data['encodings']),
                'metadata': data['metadata']
            }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
