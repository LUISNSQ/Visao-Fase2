import sys
import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO

class ControloVisao:
    def __init__(self, area_min=8000):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("A usar:", self.device)

        #yolov8n para ser mais rápido
        self.model = YOLO("yolov8n.pt")
        self.model.to(self.device)
        self.target_class = 67  # classe do telemóvel

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Erro ao abrir a câmara.")
            sys.exit(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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
        if self.before == 0:
            fps = 0
        else:
            fps = 1 / (now - self.before)
        self.before = now

        results = self.model.predict(
            frame,
            verbose=False,
            device=str(self.device),
            classes=[self.target_class],
            imgsz=480
        )[0]

        box_definida = None
        area_encontrada = 0
        pos_norm = None

        for box in results.boxes:

            classe = int(box.cls[0].item())
            if classe != self.target_class:
                continue

            conf = float(box.conf[0].item())
            if conf < 0.6:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            area = (x2 - x1) * (y2 - y1)
            if area > self.area_min and area > area_encontrada:
                area_encontrada = area
                box_definida = (x1, y1, x2, y2)

        img = frame.copy()

        if box_definida is not None:
            x1, y1, x2, y2 = box_definida

            #retangulo
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            #centro
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(img, (cx, cy), 5, (0, 255, 255), -1)

            pos_norm = cx / w

        cv2.putText(img, f"{fps:.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Camera", img)
        cv2.waitKey(1)

        return pos_norm

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
