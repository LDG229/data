import os
import torch
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from net import *
from net import *
from utils import keep_image_size_open
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# 定义图像预处理转换（保持原始尺寸，仅转换为张量）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def select_file(title, filetypes):
    """打开文件选择对话框"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return file_path


def main():
    print("欢迎使用U-Net图像分割与轮廓对齐工具")

    # 加载模型
    net = UNet().cpu()
    weights = '../params/model_weights.pth'
    if not os.path.exists(weights):
        print("错误：权重文件不存在")
        return

    net.load_state_dict(torch.load(weights, map_location='cpu'))
    net.eval()
    print("模型加载成功")

    # 选择原始图像
    test_image_path = select_file("选择原始图像", [("图像文件", "*.png;*.jpg;*.jpeg")])
    if not test_image_path:
        print("未选择图像，程序退出")
        return

    try:
        # 读取原始图像（保持原始尺寸）
        original_image = cv2.imread(test_image_path)
        if original_image is None:
            raise FileNotFoundError(f"无法读取图像：{test_image_path}")

        # 模型输入预处理（保持原始比例，填充至256x256）
        h, w = original_image.shape[:2]
        target_size = (256, 256)
        scale = min(target_size[0] / w, target_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        padded_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        x_offset = (target_size[0] - new_w) // 2
        y_offset = (target_size[1] - new_h) // 2
        padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = cv2.resize(original_image, (new_w, new_h))

        # 模型推理
        img_data = transform(padded_image).unsqueeze(0).cpu()
        with torch.no_grad():
            out = net(img_data)

        # 保存轮廓图像（模型输出为256x256，无需调整尺寸）
        os.makedirs('result', exist_ok=True)
        contour_path = 'result/test.png'
        save_image(out, contour_path)
        print(f"轮廓图像已保存至：{contour_path}")

        # 提取轮廓并绘制到原始图像
        contour_image = cv2.imread(contour_path)
        hsv = cv2.cvtColor(contour_image, cv2.COLOR_BGR2HSV)

        # 红色轮廓检测（根据模型输出颜色调整范围）
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 将轮廓坐标映射回原始图像尺寸
        scaled_contours = []
        for cnt in contours:
            cnt = cnt.astype(np.float32)
            cnt[:, :, 0] = (cnt[:, :, 0] - x_offset) / scale  # 还原x坐标
            cnt[:, :, 1] = (cnt[:, :, 1] - y_offset) / scale  # 还原y坐标
            scaled_contours.append(cnt.astype(np.int32))

        # 在原始图像上绘制轮廓
        result_image = original_image.copy()
        cv2.drawContours(result_image, scaled_contours, -1, (0, 0, 255), 4)
        result_path = 'result/aligned_contours.jpg'
        cv2.imwrite(result_path, result_image)
        print(f"最终结果已保存至：{result_path}")

        # 组合原始图像和处理后的图像
        original_img = Image.open(test_image_path)
        result_img = Image.open(result_path)
        combined_img = Image.new('RGB', (original_img.width + result_img.width, original_img.height))
        combined_img.paste(original_img, (0, 0))
        combined_img.paste(result_img, (original_img.width, 0))
        combined_path = 'result/combined_result.jpg'
        combined_img.save(combined_path)
        print(f"组合结果已保存至：{combined_path}")

    except Exception as e:
        print(f"处理失败：{str(e)}")


if __name__ == "__main__":
    main()