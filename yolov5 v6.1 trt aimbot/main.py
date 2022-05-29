import os
import sys
import time
from pathlib import Path
import numpy as np
# import serial
import torch
import win32api
import win32con

from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_img_size, cv2, non_max_suppression, xyxy2xywh, scale_coords)
from utils.torch_utils import select_device, time_sync
from grabscreen import grab_screen
from PID import PID

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

com_text = ""
configs_dict = {}
config_list = []
with open('configs.txt', 'r', encoding="utf-8") as f:
    for config_line in f:
        config_list.append(list(config_line.strip('\n').split(',')))
f.close()
config_list.remove(['# 范围调节'])
config_list.remove(['# PID控制调节'])
for i in range(10):
    config_list.remove([''])
config_list.remove([''])
index1 = config_list[0][0].find("=")
index2 = config_list[0][0].find("#")
# com参数
com_text = config_list[0][0][index1 + 1:index2].strip()
del config_list[0]
# 1-9参数
print(com_text)
print("配置读取如下\n*************************************************")
last_configs_list = []
for i in range(len(config_list)):
    index1 = config_list[i][0].find("=")
    index2 = config_list[i][0].find("#")
    last_configs_list.append(float(config_list[i][0][index1 + 1:index2]))
    configs_dict[i + 1] = float(config_list[i][0][index1 + 1:index2])

print(f"配置写入：{configs_dict}")

# ser = serial.Serial(f'{com_text}', 115200)
# ser.write('import km\r\n'.encode('utf-8'))

time.sleep(0.1)

# print('kmbox 成功导入模块:', str(ser.read(ser.inWaiting()), 'utf-8'))


y_correction_factor = configs_dict[1]  # 截图位置修正， 值越大截图窗口向上
x_correction_factor = 0  # 截图位置修正， 值越大截图窗口向右移动
screen_x, screen_y = configs_dict[2], configs_dict[3]  # 电脑显示器分辨率
window_x, window_y = configs_dict[4], configs_dict[5]  # x,y 截图窗口大小
screen_x_center = screen_x / 2
screen_y_center = screen_y / 2
PID_time = configs_dict[6]
Kp = configs_dict[7]
Ki = configs_dict[8]
Kd = configs_dict[9]
y_portion = configs_dict[10]  # 数值越小，越往下， 从上往下头的比例距离
max_step = configs_dict[11]  # 每次位移的最大步长
pid = PID(PID_time, max_step, -max_step, Kp, Ki, Kd)

grab_window_location = (
    int(screen_x_center - window_x / 2 + x_correction_factor),
    int(screen_y_center - window_y / 2 - y_correction_factor),
    int(screen_x_center + window_x / 2 + x_correction_factor),
    int(screen_y_center + window_y / 2 - y_correction_factor))

edge_x = screen_x_center - window_x / 2
edge_y = screen_y_center - window_y / 2

# 自瞄范围设置
aim_x = configs_dict[13]  # aim width
aim_x_left = int(screen_x_center - aim_x / 2)  # 自瞄左右侧边距
aim_x_right = int(screen_x_center + aim_x / 2)

aim_y = configs_dict[14]  # aim width
aim_y_up = int(screen_y_center - aim_y / 2 - y_correction_factor)  # 自瞄上下侧边距
aim_y_down = int(screen_y_center + aim_y / 2 - y_correction_factor)
time.sleep(2)


@torch.no_grad()  # 不要删 (do not delete it )
def find_target(
        weights=ROOT / 'apex_best_2.engine',  # model.pt path(s) 选择自己的模型
        # weights=ROOT / 'apex_best_2.pt',  # model.pt path(s)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=10,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    bs = 1  # batch_size
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))
    time.sleep(0.5)

    t1 = time_sync()

    # img0 = cv2.imread('./data/images/apex_test4.jpg')   # test picture
    # img0 = cv2.imread('./data/images/0.png')

    # for i in range(500):           # for i in range(500) 运行500轮测速 (run 500 rounds to check each round spend)
    print(f"imgz = {imgsz}")
    while True:
        img0 = grab_screen(grab_window_location)
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGRA2BGR)

        img = letterbox(img0, imgsz, stride=stride, auto=pt)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        # Inference
        pred = model(img, augment=False, visualize=False)
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        det = pred[0]

        target_distance_list = []
        target_xywh_list = []
        if len(det):
            # print('move 回码：', str(ser.read(ser.inWaiting()), 'utf-8'))

            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # 不使用归一化，返回坐标图片

                # print('\033[0;40;40m' + f'   xywh = {xywh}   \n' + '\033[0m')

                target_xywh_list.append(xywh)
                target_distance = abs(edge_x + xywh[0] - screen_x_center)

                target_distance_list.append(target_distance)
            # print(f"target_distance_list= {target_distance_list}")
            min_index = target_distance_list.index(min(target_distance_list))
            target_xywh = target_xywh_list[min_index]

            target_xywh_x = target_xywh[0] + edge_x
            target_xywh_y = target_xywh[1] + edge_y

            print('\033[0;33;40m' + f"target-X = {target_xywh_x}  target—Y = {target_xywh_y}" + '\033[0m')

            if aim_x_left < target_xywh_x < aim_x_right and aim_y_up < target_xywh_y < aim_y_down:

                if configs_dict[12] == 3:
                    aim_mouse = win32api.GetAsyncKeyState(win32con.VK_RBUTTON) \
                                or win32api.GetAsyncKeyState(win32con.VK_LBUTTON)
                elif configs_dict[12] == 2:
                    aim_mouse = win32api.GetAsyncKeyState(win32con.VK_RBUTTON)

                elif configs_dict[12] == 1:
                    aim_mouse = win32api.GetAsyncKeyState(win32con.VK_LBUTTON)

                else:
                    print("请填入正确的鼠标瞄准模式数字 1 或 2 或 3, Please fill the correct aim mod number 1 or 2 or 3")
                    break

                if aim_mouse:
                    # 鼠标计算相对移动距离 (calculate mouse relative move distance)
                    final_x = target_xywh_x - screen_x_center
                    final_y = target_xywh_y - screen_y_center - y_portion * target_xywh[3]

                    pid_x = int(pid.calculate(final_x, 0))
                    pid_y = int(pid.calculate(final_y, 0))
                    print(f"Mouse-Move X Y = ({pid_x}, {pid_y})")

                    """ 单片机执行位移，每个人位移的实现不一样，位移坐标你都拿到了，动鼠标的事情自己考虑
                    since you have gotten the x y movement data,choose your own way to move the mouse to aim enemy"""

                    # ser.write(f'km.move({pid_x},{pid_y})\r\n'.encode('utf-8'))




        else:
            print('\033[0;31;40m' + f'  no target   ' + '\033[0m')

    t3 = time_sync()

    # Print time (total circle)
    LOGGER.info(f'\ntime = {(t3 - t1) * 1000 / 500:.3f}ms '
                f'\n frequency = {1 / ((t3 - t1) / 500) :.3f} round per second')


if __name__ == "__main__":
    find_target()
