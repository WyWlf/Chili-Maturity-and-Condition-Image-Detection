from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')
    model.train(data='qq.v2i.yolov8\data.yaml' , epochs=100, imgsz=640, device=0)

if __name__ == '__main__':
    main()