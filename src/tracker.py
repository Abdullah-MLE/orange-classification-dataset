import numpy as np


class OrangeTracker:
    def __init__(self, max_distance, min_frames_gone):
        self.tracked_oranges = {}
        self.next_id = 1
        self.max_distance = max_distance
        self.min_frames_gone = min_frames_gone
    
    def update(self, detections):
        current_centers = []
        current_boxes = []
        
        # Get centers of current detections
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            current_centers.append((center_x, center_y))
            current_boxes.append((x1, y1, x2, y2))
        
        # Update tracked oranges
        self._update_tracked(current_centers, current_boxes)
        
        # Remove old oranges
        self._remove_old_oranges()
        
        return self.tracked_oranges
    
    def _update_tracked(self, current_centers, current_boxes):
        # Mark all as not seen
        for orange_id in self.tracked_oranges:
            self.tracked_oranges[orange_id]['seen'] = False
        
        # Match current detections to tracked oranges
        used_indices = set()
        
        for i, center in enumerate(current_centers):
            matched_id = self._find_match(center)
            
            if matched_id is not None:
                # Update existing orange
                self.tracked_oranges[matched_id]['center'] = center
                self.tracked_oranges[matched_id]['bbox'] = current_boxes[i]
                self.tracked_oranges[matched_id]['seen'] = True
                self.tracked_oranges[matched_id]['frames_missing'] = 0
                used_indices.add(i)
            else:
                # New orange
                orange_id = self.next_id
                self.next_id += 1
                
                self.tracked_oranges[orange_id] = {
                    'center': center,
                    'bbox': current_boxes[i],
                    'seen': True,
                    'frames_missing': 0,
                    'classified': False,
                    'class_name': None,
                    'quality_value': None
                }
    
    def _find_match(self, center):
        min_dist = self.max_distance
        matched_id = None
        
        for orange_id, data in self.tracked_oranges.items():
            if not data['seen']:
                dist = self._calculate_distance(center, data['center'])
                if dist < min_dist:
                    min_dist = dist
                    matched_id = orange_id
        
        return matched_id
    
    def _calculate_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _remove_old_oranges(self):
        to_remove = []
        
        for orange_id, data in self.tracked_oranges.items():
            if not data['seen']:
                data['frames_missing'] += 1
                
                if data['frames_missing'] > self.min_frames_gone:
                    to_remove.append(orange_id)
        
        for orange_id in to_remove:
            del self.tracked_oranges[orange_id]
    
    def is_new_orange(self, orange_id):
        return not self.tracked_oranges[orange_id]['classified']
    
    def mark_classified(self, orange_id, class_name, quality_value):
        self.tracked_oranges[orange_id]['classified'] = True
        self.tracked_oranges[orange_id]['class_name'] = class_name
        self.tracked_oranges[orange_id]['quality_value'] = quality_value