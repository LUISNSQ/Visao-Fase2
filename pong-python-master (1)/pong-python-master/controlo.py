import sys
import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO

class ControloVisao:
    def __init__(self, area_min=8000):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.model = YOLO("yolov8s.pt")
        self.model.to(self.device)
        self.target_class = 67 # a classe 67 é a classe do telemovel para a lista Ultralitics yolov8

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Erro ao abrir a câmara.")
            sys.exit(1)

        self.area_min = area_min
        self.before = 0

        cv2.namedWindow("Camera")

    def detetor(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        #Calculo dos FPS
        now = time.time()
        fps = 1 / (now - self.before)
        self.before = now

        results = self.model.predict(frame,verbose=False,device=str(self.device) )[0]
        box_definida = None
        area_encontrada = 0
        pos_norm = None

        for box in results.boxes:
            classe = int(box.cls[0]) # verifica a classe do COCO que encontrou
            if classe != self.target_class:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)

            if area > self.area_min and area > area_encontrada:
                area_encontrada = area
                box_definida = (x1, y1, x2, y2)

        img = frame.copy()

        if box_definida is not None:
            x1, y1, x2, y2 = box_definida

            # borda
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # centro
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)

            pos_norm = cx / w

        cv2.putText(img, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Camera", img)
        cv2.waitKey(1)

        return pos_norm

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
