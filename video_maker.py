import cv2
import os

image_folder = './dataset/hotdog/40_frames/'
output_video = './hotdog.mp4'

# 获取目录中的所有图像文件名
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
# 按照文件名排序
images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))# r_23.png

# 读取第一张图像来获取帧的宽和高
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# 定义视频编码器和输出文件格式
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'XVID', 'MJPG', 'X264', 等
video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

# 遍历图像文件，将每张图像写入视频
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

video.release()
print(f"video saved in {output_video}")
