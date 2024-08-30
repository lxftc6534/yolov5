import torch
import cv2
import pyautogui
from PIL import Image, ImageGrab
import time
import numpy as np

# 加载模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r"runs/train/exp13/weights/best.pt")

# 捕获游戏画面
def capture_game_screenshot():
    screenshot = pyautogui.screenshot()
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screenshot

def get_screenshot():
    im = ImageGrab.grab()
    imm = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    imm = imm[0:1000, 0:1000]
    # imm = cv2.resize(imm, None, fx=0.5, fy=0.5)
    imm = cv2.resize(imm, None, fx=1, fy=1)
    return imm

# 检测对象并返回结果
def detect_objects(img):
    results = model(img)
    return results

# 执行相应的游戏操作
def perform_action(detections):
    for detection in detections:
        print("detection", detection)
        # 检测到的对象坐标和标签
        x1, y1, x2, y2, confidence, class_id = detection
        # 根据检测对象执行操作，如攻击怪物
        if class_id == 0:  # 假设0是怪物
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            print(f"click({center_x},{center_y})")
            pyautogui.click(center_x, center_y)  # 模拟点击怪物位置
            pyautogui.press('space')  # 模拟攻击键
            pyautogui.press('T')  # 模拟攻击键


if __name__ == "__main__":
    # 主循环
    while True:
        # screenshot = capture_game_screenshot()
        screenshot = get_screenshot()
        # cv2.imshow("screenshot", screenshot)
        # im2 = "images/test/Apple2.jpeg"
        # results = detect_objects(im2)
        results = detect_objects(screenshot)
        xyxy = results.xyxy[0]
        print('xyxy', xyxy)
        detections = xyxy.numpy()
        print('detections', detections)
        perform_action(detections)
        # time.sleep(1)  # 控制脚本运行速度
        # cv2.imshow("capture", screenshot)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # q键推出
            break

