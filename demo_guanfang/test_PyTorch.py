import base64
from io import BytesIO

import cv2
import torch
from PIL import Image, ImageGrab
import threading
from ultralytics import YOLO

def demo1():
    # Model
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")

    # Image
    im = "https://ultralytics.com/images/zidane.jpg"

    # Inference
    results = model(im)

    results.pandas().xyxy[0] # Pandas DataFrame
    #      xmin    ymin    xmax   ymax  confidence  class    name
    # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
    # 1  433.50  433.50   517.5  714.5    0.687988     27     tie
    # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
    # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
    results.save()
    return results

def test_results():
    # Model
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")

    # Images
    for f in "zidane.jpg", "bus.jpg":
        torch.hub.download_url_to_file("https://ultralytics.com/images/" + f, f"download/{f}")  # download 2 images
    im1 = Image.open("download/zidane.jpg")  # PIL image
    im2 = cv2.imread("download/bus.jpg")[..., ::-1]  # OpenCV image (BGR to RGB)

    # Inference
    results = model([im1, im2], size=640)  # batch of images

    # Results
    results.print()
    results.save()  # or .show() 大图，包含识别出的标识
    # results.show()  # or .show()

    print('results.xyxy[0]', results.xyxy[0])
    print('results.pandas().xyxy[0] ', results.pandas().xyxy[0])
    # results.xyxy[0]  # im1 predictions (tensor)
    # results.pandas().xyxy[0]  # im1 predictions (pandas)
    #      xmin    ymin    xmax   ymax  confidence  class    name
    # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
    # 1  433.50  433.50   517.5  714.5    0.687988     27     tie
    # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
    # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
    return results


def getbase64_by_results(results):
    ## 获取base64
    results.ims  # array of original images (as np array) passed to model for inference
    results.render()  # updates results.ims with boxes and labels
    for im in results.ims:
        buffered = BytesIO()
        im_base64 = Image.fromarray(im)
        im_base64.save(buffered, format="JPEG")
        print(base64.b64encode(buffered.getvalue()).decode("utf-8"))  # base64 encoded image with results


def test_crop_by_results(results):
    crops = results.crop(save=True)  # cropped detections dictionary


def test_sort_by_results(results):
    # results.pandas().xyxy[0].sort_values("xmin")  # sorted left-right
    print(r'{results.pandas().xyxy[0].sort_values("xmin")}')
    print(results.pandas().xyxy[0].sort_values("xmin"))


def test_to_json_by_results(results):
    # results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    print(results.pandas().xyxy[0].to_json(orient="records"))

def test_ImageGrab():
    # Model
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")

    # Image
    im = ImageGrab.grab()  # take a screenshot

    # Inference
    results = model(im)
    # Results
    results.print()
    results.save()  # or .show()


def test_custom():
    # 测试成功（官方模型）
    # model = torch.hub.load("ultralytics/yolov5", "custom", path="weights/best.pt")  # local model
    # 测试成功（使用本地yolov5）
    # model = torch.hub.load(r"C:\work\python\yolov5", "custom", path=r"C:\work\python\yolov5\demo1\runs\train\exp2\weights\best.pt", source="local")  # local repo
    # 测试成功（通过本地直接运行文件直接训练的best.pt）
    # model = torch.hub.load("ultralytics/yolov5", "custom", path=r"C:\work\python\yolov5\demo1\runs\train\exp2\weights\best.pt")  # local repo
    # 测试成功（通过本地命令训练的best.pt： demo_guanfang/train.py --img 640 --epochs 100 --data coco128.yaml --weights yolov5s.pt）
    # model = torch.hub.load("ultralytics/yolov5", "custom", path=r"runs/train/exp2/weights/best.pt")  # local repo
    # 测试成功（通过本地命令训练的best.pt： demo_guanfang/train.py --img 640 --epochs 100 --data coco128.yaml）
    model = torch.hub.load("ultralytics/yolov5", "custom", path=r"runs/train/exp5/weights/best.pt")  # local repo
    # model = YOLO("runs/train/exp/weights/best.pt")  # local repo
    im = "images/test/txsp.png"
    results = model(im)
    results.pandas().xyxy[0]  # Pandas DataFrame
    results.save()

def run(model, im):
    """Performs inference on an image using a given model and saves the output; model must support `.save()` method."""
    results = model(im)
    results.save()


def test_Thread():
    # Models
    model0 = torch.hub.load("ultralytics/yolov5", "yolov5s", device=0)
    model1 = torch.hub.load("ultralytics/yolov5", "yolov5s", device=1)

    # Inference
    threading.Thread(target=run, args=[model0, "https://ultralytics.com/images/zidane.jpg"], daemon=True).start()
    threading.Thread(target=run, args=[model1, "https://ultralytics.com/images/bus.jpg"], daemon=True).start()


if __name__ == "__main__":
    # test_to_json_by_results(demo1())
    test_custom()
