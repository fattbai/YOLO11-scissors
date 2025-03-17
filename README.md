# YOLO11-scissors
这是一个YOLO改进型，将使用scissors处理后的梯度图像作为辅助信息连同原始图像一起输入模型

### 使用
本项目基于YOLO实现
[https://github.com/ultralytics/ultralytics]

0.按照YOLO官方的指引安装好YOLO环境以及YOLO代码\n
1.将本仓库中的yolo11n-scissors.yaml放入ultralytics-main/ultralytics/cfg/models/11文件夹\n
2.将本仓库中的scissors.py放入ultralytics-main/ultralytics/nn/modules文件夹\n
3.使用本仓库中的tasks.py文件替换掉ultralytics-main/ultralytics/nn/tasks.py文件\n
4.在ultralytics-main环境下，运行以下代码：

```
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n-scissors.yaml").load("yolo11n.pt")

    results = model.train(data="coco.yaml", imgsz=640, epochs=300, optimizer="SGD", lr0=0.01, lrf=0.0001)
```
