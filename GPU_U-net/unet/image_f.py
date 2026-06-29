from PIL import Image
import os

def process_images(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否为图片文件
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            file_path = os.path.join(folder_path, filename)
            try:
                # 打开图片
                image = Image.open(file_path)

                # 左转
                left_rotated = image.rotate(90, expand=True)
                left_filename = os.path.splitext(filename)[0] + 'l' + os.path.splitext(filename)[1]
                left_file_path = os.path.join(folder_path, left_filename)
                left_rotated.save(left_file_path)

                # 右转
                right_rotated = image.rotate(-90, expand=True)
                right_filename = os.path.splitext(filename)[0] + 'r' + os.path.splitext(filename)[1]
                right_file_path = os.path.join(folder_path, right_filename)
                right_rotated.save(right_file_path)

                # 上下翻转
                flipped = image.transpose(Image.FLIP_TOP_BOTTOM)
                flipped_filename = os.path.splitext(filename)[0] + 'u' + os.path.splitext(filename)[1]
                flipped_file_path = os.path.join(folder_path, flipped_filename)
                flipped.save(flipped_file_path)

                print(f"处理完成: {filename}")
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")

# 指定文件夹路径
folder_path = 'E:\GPU_U-net\image_use\mask'  # 替换为实际的文件夹路径
process_images(folder_path)