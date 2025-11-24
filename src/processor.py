import cv2
from .config import config


class FrameProcessor:
    def __init__(self, models, tracker, queue_manager):
        self.models = models
        self.tracker = tracker
        self.queue = queue_manager
    
    def process_frame(self, frame):
        # Detect oranges
        detections = self.models.detect(frame, config.DETECTION_CONF)
        
        if len(detections[0].boxes) == 0:
            self.tracker.update([])
            return frame
        
        # Update tracker
        tracked = self.tracker.update(detections[0].boxes)
        
        # Process each tracked orange
        for orange_id, data in tracked.items():
            # Only classify new oranges
            if self.tracker.is_new_orange(orange_id):
                self._classify_and_queue(frame, orange_id, data)
            
            # Draw on frame
            self._draw_orange(frame, data, orange_id)
        
        return frame
    
    def _classify_and_queue(self, frame, orange_id, data):
        # Crop orange
        crop = self._extract_crop(frame, data['bbox'])
        
        if crop.size == 0:
            return
        
        # Classify
        class_name, confidence = self.models.classify(crop)
        
        if class_name is None or confidence < config.CLASSIFICATION_CONF:
            return
        
        # Determine quality
        quality_value = config.FRESH_VALUE if 'FRESH' in class_name else config.ROTTEN_VALUE
        
        # Mark as classified
        self.tracker.mark_classified(orange_id, class_name, quality_value)
        
        # Add to queue
        print(f"\nOrange #{orange_id}: {class_name} (conf: {confidence:.2f})")
        self.queue.add(quality_value)
    
    def _extract_crop(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        
        # Add padding
        padding = 15
        h, w = frame.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        crop = frame[y1:y2, x1:x2]
        
        # Resize
        if crop.size > 0:
            crop = cv2.resize(crop, (224, 224))
        
        return crop
    
    def _draw_orange(self, frame, data, orange_id):
        x1, y1, x2, y2 = data['bbox']
        
        # Color based on classification
        if data['classified']:
            color = (0, 255, 0) if data['quality_value'] == 1 else (0, 0, 255)
            label = f"#{orange_id}: {data['class_name']}"
        else:
            color = (255, 165, 0)
            label = f"#{orange_id}: Detecting..."
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)