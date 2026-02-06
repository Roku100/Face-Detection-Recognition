import numpy as np

class FaceMatcher:
    def __init__(self, config, database):
        self.config = config
        self.database = database
        self.tolerance = config.get('recognition.tolerance', 0.6)
    
    def _calculate_distance(self, encoding1, encoding2):
        # Chi-square distance for histogram comparison
        epsilon = 1e-10
        distance = 0.5 * np.sum(((encoding1 - encoding2) ** 2) / (encoding1 + encoding2 + epsilon))
        return distance
    
    def _calculate_distances(self, encoding, known_encodings):
        distances = []
        for known_encoding in known_encodings:
            distance = self._calculate_distance(encoding, known_encoding)
            distances.append(distance)
        return np.array(distances)
    
    def match_face(self, encoding):
        known_encodings, known_names = self.database.get_all_encodings()
        
        if not known_encodings:
            return None, 0.0
        
        distances = self._calculate_distances(encoding, known_encodings)
        best_match_idx = np.argmin(distances)
        best_distance = distances[best_match_idx]
        
        if best_distance <= self.tolerance:
            name = known_names[best_match_idx]
            confidence = 1.0 - (best_distance / 2.0)
            confidence = max(0.0, min(1.0, confidence))
            return name, confidence
        
        return None, 0.0
    
    def match_faces(self, encodings):
        results = []
        for encoding in encodings:
            name, confidence = self.match_face(encoding)
            results.append((name, confidence))
        return results
    
    def match_with_voting(self, encoding, top_k=3):
        known_encodings, known_names = self.database.get_all_encodings()
        
        if not known_encodings:
            return None, 0.0
        
        distances = self._calculate_distances(encoding, known_encodings)
        top_k_indices = np.argsort(distances)[:top_k]
        top_k_names = [known_names[i] for i in top_k_indices]
        top_k_distances = distances[top_k_indices]
        
        valid_matches = [(name, dist) for name, dist in zip(top_k_names, top_k_distances) 
                        if dist <= self.tolerance]
        
        if not valid_matches:
            return None, 0.0
        
        name_votes = {}
        for name, dist in valid_matches:
            if name not in name_votes:
                name_votes[name] = []
            confidence = 1.0 - (dist / 2.0)
            confidence = max(0.0, min(1.0, confidence))
            name_votes[name].append(confidence)
        
        best_name = max(name_votes.items(), key=lambda x: np.mean(x[1]))
        avg_confidence = np.mean(best_name[1])
        
        return best_name[0], avg_confidence
    
    def get_all_matches(self, encoding, threshold=None):
        if threshold is None:
            threshold = self.tolerance
        
        known_encodings, known_names = self.database.get_all_encodings()
        
        if not known_encodings:
            return []
        
        distances = self._calculate_distances(encoding, known_encodings)
        
        matches = []
        for name, distance in zip(known_names, distances):
            if distance <= threshold:
                confidence = 1.0 - (distance / 2.0)
                confidence = max(0.0, min(1.0, confidence))
                matches.append((name, confidence))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
