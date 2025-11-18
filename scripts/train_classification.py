import os
from ultralytics import YOLO


def train_classification_model():
    """
    Train a YOLOv8 classification model using the dataset already available
    inside the project at: orange_classification_dataset/
    The trained model will be saved automatically inside: models/
    """

    # Path to the dataset inside the project
    dataset_path = os.path.join("..", "orange_classification_dataset")

    # Ensure the dataset exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found at: {dataset_path}")

    # Load classification model
    model = YOLO("yolov8n-cls.pt")

    # Train the model
    results = model.train(
        data=dataset_path,           # Path containing train/ and val/
        epochs=20,                   # You can change this
        imgsz=160,                   # Classification recommended size
        batch=32,
        name="orange_classifier",    # Run name inside runs/classify/
        project="models",            # Save model in models/
        save=True,
        exist_ok=True,
        workers=0
    )

    print("Training completed.")
    print("Model weights saved inside: models/orange_classifier/")


if __name__ == "__main__":
    train_classification_model()
