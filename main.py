from ultralytics import YOLO

#Load the model
model = YOLO('yolov8m.yaml')

result = model.train(data='config.yaml', epochs=30)

metrics = model.val()  # evaluate model performance on the validation set
results = model("F:/AI ML DL Projects/Supports/Yolo object detection/dataset/data/images/val/134.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format