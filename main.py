import cv2
from src.config import config
from src.models import ModelLoader
from src.tracker import OrangeTracker
from src.queue_manager import QualityQueue
from src.processor import FrameProcessor


def main():
    # Initialize components
    models = ModelLoader(config.DETECTION_MODEL, config.CLASSIFICATION_MODEL)
    tracker = OrangeTracker(config.MAX_DISTANCE, config.MIN_FRAMES_GONE)
    queue_manager = QualityQueue()
    processor = FrameProcessor(models, tracker, queue_manager)
    
    # Open camera
    cap = cv2.VideoCapture(config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        return
    
    print("\n" + "="*50)
    print("Orange Quality Control System Started")
    print("="*50)
    print("Press 'q' to quit\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Cannot read frame!")
                break
            
            # Flip frame
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame = processor.process_frame(frame)
            
            # Draw stats
            stats = queue_manager.get_stats()
            cv2.putText(processed_frame, f"Queue: {stats['queue_size']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Fresh: {stats['total_fresh']}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Rotten: {stats['total_rotten']}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show frame
            if config.SHOW_WINDOW:
                cv2.imshow('Orange Quality Control', processed_frame)
            
            # Check quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final stats
        stats = queue_manager.get_stats()
        print("\n" + "="*50)
        print("Final Statistics:")
        print("="*50)
        print(f"Total oranges processed: {stats['total']}")
        print(f"Fresh oranges: {stats['total_fresh']}")
        print(f"Rotten oranges: {stats['total_rotten']}")
        print(f"Queue size: {stats['queue_size']}")
        print("="*50)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()