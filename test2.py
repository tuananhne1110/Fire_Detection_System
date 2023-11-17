from ultralytics import YOLO
model = YOLO('D:\\Fire_Detection_System\\yolo\\train\\Weights\\best.pt')
#model.predict(source ='D:\\Red\\Red\\ivan-torres-AXz346Rhs6A-unsplash.jpg',imgsz = 640, conf = 0.2, save = True)
model.predict(source = 0, imgsz= 640,conf = 0.5,save = False,show = True)