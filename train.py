from ultralytics import YOLO

def main():
    model = YOLO('chili.pt')
    model.train(data="qq.v2i.yolov8\data.yaml", epochs=100, imgsz=640)
if __name__ == '__main__':
    main()