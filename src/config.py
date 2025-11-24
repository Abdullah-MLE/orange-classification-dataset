class Config:
    # Model paths
    DETECTION_MODEL = 'models/classification/best.pt'
    CLASSIFICATION_MODEL = 'models/object_detection/orange_fresh_rotten_classifier_v2.pt'
    
    # Camera
    CAMERA_ID = 'old\\dataset\\output_video.mp4'  # Change to 0 for webcam
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    
    # Detection/Classification
    DETECTION_CONF = 0.4
    CLASSIFICATION_CONF = 0.5
    
    # Tracking
    MAX_DISTANCE = 100  # Max pixel distance to consider same orange
    MIN_FRAMES_GONE = 15  # Frames before considering orange left
    
    # Queue values
    FRESH_VALUE = 1
    ROTTEN_VALUE = 0
    
    # Display
    SHOW_WINDOW = True


config = Config()