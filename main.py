from ultralytics import YOLO

model = YOLO("yolov8n.yaml") # Build a new model from scratch

# Use the model
results = model.train(data="dataset.yaml", epochs=5) # Train the model