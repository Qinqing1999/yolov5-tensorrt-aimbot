# yolov5-tensorrt-aimbot
FPS game AI aimassist framework
AI 是拿来用的，不是用来坑小白钱的

框架说明:

需要有一点点python基础，完全没有编程基础建议学半个月再来

此ai框架支持.pt 和 .trt 两种模型（yolov5 6.1类里的DetectMultiBackend函数自带自动识别并加载各种模型，本来还支持onnx的，但是我不想再下载TensorFlow了）,Pt模型应该是人人都能用，trt需要根据自己的显卡编译一次（编译方法最后有官方链接）。

模型输入维度是640*640 ，你可以换自己的权重，如果用自己的网络模型输入尺寸不是640*640，修改imgz = (640, 640)即可，模型需要fp16，如果不是fp16模型关闭half即可（half = False），尽量使用半精度模型，速度更快，而且精度没什么丢失。

修改瞄准的参数都在config.txt文档里，config.txt文档不要改顺序，也不加空行（主要是读取txt的函数写得有点麻瓜）

main.py注释掉的如 # ser.write(f'km.move({pid_x},{pid_y})\r\n'.encode('utf-8')) 等语句是我用的kmbox位移鼠标的语句。你都拿到相对位移坐标了，自己加一个鼠标位移就行

安装说明（一定按以下顺序安装环境依赖包）
1. 安装anaconda 或者 mini conda
2. conda 创建虚拟环境
3.启动虚拟环境 
conda activate <your env name>
4.如果是中国大陆用户切换清华源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
5.安装cuda、cudnn （必须先安装cuda 和cudnn）
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
6.安装tensorrt：
去英伟达官网下载TensorRT-8.2.1.8.Windows10.x86_64.安装包（只要是支持的版本都可以），解压后cd切换到 TensorRT-8.2.1.8\python文件夹, 使用pip install命令安装对应python版本的.whl文件（cp39对应python3.9，cp36对应python3.6）

示例:  pip install tensorrt-8.2.1.8-cp39-none-win_amd64.whl 
7.安装yolov5 version6.1依赖:
pip install -r requirements.txt

8.完成上面安装后，最后安装win32 库（可能会报错，网上有解决方法，很容易解决）:
pip install pypiwin32 
或者pip install pywin32 
9.运行main.py 即可
可以打开apex_test.jpg的图片，坐面拖动图片测试，正常情况扫识别到目标应该会打印坐标
补充
Pt模型转trt模型 ，yolov5 github官方有详细说明：
https://github.com/ultralytics/yolov5/issues/251   
下载yolov5 6.1，用上面的环境编译即可


开源by：山高水长流
