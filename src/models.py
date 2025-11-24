from ultralytics import YOLO
import cv2


class ModelLoader:
    def __init__(self, detection_path, classification_path):
        print("Loading models...")
        self.detector = YOLO(detection_path)
        self.classifier = YOLO(classification_path)
        print(f"Models loaded!")
        print(f"Classifier classes: {self.classifier.names}")
    
    def detect(self, frame, conf):
        return self.detector.predict(frame, conf=conf, verbose=False)
    
    def classify(self, crop):
        if crop.size == 0:
            return None, 0.0
        
        # Enhance image
        enhanced = cv2.convertScaleAbs(crop, alpha=1.2, beta=10)
        
        # Classify
        results = self.classifier(enhanced, verbose=False)
        
        if results[0].probs is not None:
            probs = results[0].probs
            class_idx = int(probs.top1)
            confidence = float(probs.top1conf)
            class_name = self.classifier.names[class_idx].upper()
            
            return class_name, confidence
        
        return None, 0.0