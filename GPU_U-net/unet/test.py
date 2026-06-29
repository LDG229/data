import os
import torch
import cv2
import numpy as np
import logging  # 新增：导入logging模块
from tkinter import Tk, Button, Label, filedialog, Frame, Checkbutton, IntVar
from torchvision import transforms
from net import UNet
from utils import keep_image_size_open, detect_circles
from torchvision.utils import save_image
from PIL import Image, ImageTk

# -------------------------- 新增：配置logging（确保日志功能可用） --------------------------
logging.basicConfig(
    level=logging.INFO,  # 日志级别：INFO及以上会被输出
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式：时间+级别+内容
    handlers=[
        logging.StreamHandler()  # 输出到控制台（也可添加FileHandler保存到文件）
    ]
)


# -------------------------- 1. 内置safe_cv2_read函数（解决中文路径读取） --------------------------
def safe_cv2_read(path):
    """安全读取图像（支持中文路径，校验完整性）"""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在：{path}")
        file_size = os.path.getsize(path)
        if file_size == 0:
            raise RuntimeError(f"文件为空（已损坏）：{path}（大小：0字节）")
        if file_size > 10 * 1024 * 1024:
            raise RuntimeError(f"文件过大（{file_size // 1024 // 1024}MB），建议压缩至10MB内")

        # 中文路径读取核心逻辑
        img_bytes = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"无法解码图像（格式不支持）：{path}")

        # 通道自动转换（适配后续预处理）
        if len(img.shape) == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 4:
            background = np.ones_like(img[:, :, :3]) * 255
            alpha = img[:, :, 3] / 255.0
            img = (background * (1 - alpha) + img[:, :, :3] * alpha).astype(np.uint8)

        logging.info(f"成功读取图像：{path}（尺寸：{img.shape[1]}×{img.shape[0]}）")
        return img
    except Exception as e:
        logging.error(f"图像读取失败：{str(e)}")
        raise RuntimeError(f"图像读取失败：{str(e)}（路径：{path}）")


# -------------------------- 2. 全局初始化 --------------------------
# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 模型初始化
net = UNet().cpu()
DEFAULT_WEIGHT_PATH = '../params/model_weights.pth'
model_loaded = False

# 路径变量
selected_image_path = None
preprocessed_image_path = None
segmented_image_path = 'result/test.png'

# Tkinter变量（延后初始化）
binary_flag = None
gaussian_flag = None

# UI组件引用
original_label = None  # 原始图像标签
preprocess_label = None  # 预处理图像标签
segmented_label = None  # 分割结果标签
overlay_label = None  # 叠加效果图标签
circle_label = None  # 圆形检测结果标签
original_text_label = None  # 原始图像文本标签
preprocess_text_label = None  # 预处理文本标签
segmented_text_label = None  # 分割结果文本标签
overlay_text_label = None  # 叠加图文本标签
circle_text_label = None  # 圆形检测文本标签
status_label = None  # 状态标签


# -------------------------- 3. 核心功能 --------------------------
def load_model():
    global net, model_loaded
    file_path = filedialog.askopenfilename(
        title="选择模型权重文件",
        filetypes=[("权重文件", "*.pth")],
        initialdir=os.path.dirname(DEFAULT_WEIGHT_PATH)
    )
    if file_path:
        try:
            net.load_state_dict(torch.load(file_path, map_location='cpu'))
            net.eval()
            model_loaded = True
            logging.info(f"模型加载成功：{file_path}")
            update_status("✅ 模型加载成功 | 操作流程：1.加载模型→2.选图→3.预处理→4.分割→5.检测圆形")
        except Exception as e:
            logging.error(f"模型加载失败：{str(e)}")
            update_status(f"❌ 模型加载失败：{str(e)} | 操作流程：1.加载模型→2.选图→3.预处理→4.分割→5.检测圆形")
    else:
        update_status("❌ 未选择模型权重 | 操作流程：1.加载模型→2.选图→3.预处理→4.分割→5.检测圆形")


def select_image():
    global selected_image_path, original_label, original_text_label, overlay_label, overlay_text_label
    # 销毁旧叠加图标签
    if overlay_label is not None:
        overlay_label.destroy()
        overlay_label = None
    if overlay_text_label is not None:
        overlay_text_label.destroy()
        overlay_text_label = None

    # 选择图像
    file_path = filedialog.askopenfilename(
        title="选择原始图像",
        filetypes=[("图像文件", "*.png;*.jpg;*.jpeg;*.bmp"), ("所有文件", "*.*")],
        initialdir=os.path.join(os.path.expanduser("~"), "Desktop")
    )
    if file_path:
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在：{file_path}")
            if os.path.getsize(file_path) == 0:
                raise RuntimeError(f"文件为空（已损坏）：{file_path}")

            selected_image_path = file_path
            # 预览原始图像
            original_img = Image.open(file_path).resize((256, 256), Image.Resampling.LANCZOS)
            original_img_tk = ImageTk.PhotoImage(original_img)

            # 销毁旧标签
            if original_label is not None:
                original_label.destroy()
            if original_text_label is not None:
                original_text_label.destroy()

            # 创建新标签
            original_text_label = Label(original_frame, text="原始图像", font=("Arial", 10, "bold"))
            original_text_label.pack(pady=5)
            original_label = Label(original_frame, image=original_img_tk)
            original_label.image = original_img_tk
            original_label.pack(padx=10, pady=10)

            short_name = os.path.basename(file_path)
            logging.info(f"已选择原始图像：{file_path}")
            update_status(f"✅ 已选择图像：{short_name} | 操作流程：1.加载模型→2.选图→3.预处理→4.分割→5.检测圆形")
        except Exception as e:
            logging.error(f"图像选择失败：{str(e)}")
            update_status(f"❌ 图像选择失败：{str(e)} | 操作流程：1.加载模型→2.选图→3.预处理→4.分割→5.检测圆形")
    else:
        selected_image_path = None
        update_status("❌ 未选择图像 | 操作流程：1.加载模型→2.选图→3.预处理→4.分割→5.检测圆形")


def apply_preprocessing():
    """预处理功能（二值化+高斯去噪）"""
    global preprocessed_image_path, preprocess_label, preprocess_text_label
    if not selected_image_path:
        update_status("❌ 预处理失败：请先选择原始图像（步骤2）")
        return
    if binary_flag is None or gaussian_flag is None:
        update_status("❌ 预处理开关未初始化")
        return

    try:
        # 读取图像（调用内置的safe_cv2_read）
        img = safe_cv2_read(selected_image_path)
        processed_img = img.copy()
        preprocess_steps = []

        # 1. 高斯去噪
        if gaussian_flag.get() == 1:
            processed_img = cv2.GaussianBlur(processed_img, ksize=(5, 5), sigmaX=0)
            preprocess_steps.append("高斯去噪")
            logging.info("已对图像应用高斯去噪（5×5核）")

        # 2. 二值化
        if binary_flag.get() == 1:
            if len(processed_img.shape) == 3:
                gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = processed_img
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            preprocess_steps.append("二值化")
            logging.info(f"已对图像应用二值化（OTSU阈值：{_}）")

        # 3. 保存预处理结果
        os.makedirs('result', exist_ok=True)
        original_filename = os.path.basename(selected_image_path)
        step_suffix = "_plus_".join(preprocess_steps) if preprocess_steps else "no_process"
        preprocessed_image_path = os.path.join(
            'result',
            f"pre_{step_suffix}_{original_filename}"
        )
        # 保存图像
        is_saved = cv2.imwrite(preprocessed_image_path, processed_img)
        if not is_saved:
            ext = os.path.splitext(preprocessed_image_path)[1].lower() or '.jpg'
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95] if ext == '.jpg' else [int(cv2.IMWRITE_PNG_COMPRESSION),
                                                                                      3]
            result, img_encode = cv2.imencode(ext, processed_img, encode_param)
            if result:
                with open(preprocessed_image_path, 'wb') as f:
                    f.write(img_encode)
            else:
                raise RuntimeError(f"预处理结果保存失败：{preprocessed_image_path}")
        logging.info(f"预处理结果已保存：{preprocessed_image_path}")

        # 4. 预览预处理结果
        processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        preview_img = Image.fromarray(processed_rgb).resize((256, 256), Image.Resampling.LANCZOS)
        preview_tk = ImageTk.PhotoImage(preview_img)

        # 销毁旧标签
        if preprocess_label is not None:
            preprocess_label.destroy()
        if preprocess_text_label is not None:
            preprocess_text_label.destroy()

        # 创建新标签
        step_text = " + ".join(preprocess_steps) if preprocess_steps else "无预处理"
        preprocess_text_label = Label(preprocess_frame, text=f"预处理后\n（{step_text}）", font=("Arial", 10, "bold"))
        preprocess_text_label.pack(pady=5)
        preprocess_label = Label(preprocess_frame, image=preview_tk)
        preprocess_label.image = preview_tk
        preprocess_label.pack(padx=10, pady=10)

        update_status(f"✅ 预处理完成：{step_text} | 结果保存至：{os.path.basename(preprocessed_image_path)}")
    except Exception as e:
        logging.error(f"预处理失败：{str(e)}")
        update_status(f"❌ 预处理失败：{str(e)} | 请检查图像路径/完整性")


def start_analysis():
    global segmented_label, segmented_text_label, overlay_label, overlay_text_label, preprocessed_image_path
    if not model_loaded:
        update_status("❌ 分割失败：请先加载模型（步骤1）")
        return
    if not selected_image_path:
        update_status("❌ 分割失败：请先选择原始图像（步骤2）")
        return

    try:
        # 读取输入图像
        input_path = preprocessed_image_path if preprocessed_image_path else selected_image_path
        img = keep_image_size_open(input_path, size=(256, 256))
        logging.info(f"分割输入图像：{input_path}（已调整为256×256）")

        # 模型推理
        img_tensor = transform(img).cpu()
        img_tensor = torch.unsqueeze(img_tensor, dim=0)
        net.eval()
        with torch.no_grad():
            seg_output = net(img_tensor)

        # 保存分割结果
        os.makedirs('result', exist_ok=True)
        save_image(seg_output, segmented_image_path)
        logging.info(f"分割结果已保存：{segmented_image_path}")

        # 预览分割结果
        seg_img = Image.open(segmented_image_path).convert('RGB').resize((256, 256))
        seg_tk = ImageTk.PhotoImage(seg_img)

        # 销毁旧标签
        if segmented_label is not None:
            segmented_label.destroy()
        if segmented_text_label is not None:
            segmented_text_label.destroy()

        # 创建新分割标签
        segmented_text_label = Label(segment_frame, text="分割结果", font=("Arial", 10, "bold"))
        segmented_text_label.pack(pady=5)
        segmented_label = Label(segment_frame, image=seg_tk)
        segmented_label.image = seg_tk
        segmented_label.pack(padx=10, pady=10)

        # 预览叠加效果图
        original_img = Image.open(selected_image_path).convert('RGB').resize((256, 256))
        overlay_img = Image.blend(original_img, seg_img, alpha=0.5)
        overlay_tk = ImageTk.PhotoImage(overlay_img)

        # 销毁旧叠加标签
        if overlay_label is not None:
            overlay_label.destroy()
        if overlay_text_label is not None:
            overlay_text_label.destroy()

        # 创建新叠加标签
        overlay_text_label = Label(overlay_frame, text="原图+分割叠加", font=("Arial", 10, "bold"))
        overlay_text_label.pack(pady=5)
        overlay_label = Label(overlay_frame, image=overlay_tk)
        overlay_label.image = overlay_tk
        overlay_label.pack(padx=10, pady=10)

        logging.info("分割完成，已生成叠加效果图")
        update_status("✅ 分割完成 | 操作流程：1.加载模型→2.选图→3.预处理→4.分割→5.检测圆形")
    except Exception as e:
        logging.error(f"分割失败：{str(e)}")
        update_status(f"❌ 分割失败：{str(e)} | 请检查模型或图像")


def detect_circles_and_save_count():
    global circle_label, circle_text_label
    if not os.path.exists(segmented_image_path):
        update_status("❌ 圆形检测失败：请先执行分割（步骤4）")
        return

    try:
        circle_count = detect_circles(segmented_image_path)
        logging.info(f"圆形检测完成：共{circle_count}个有效圆形")
        result_text = f"圆形检测结果\n共检测到 {circle_count} 个有效圆形\n详情保存至：result/circle_data.xlsx"

        # 销毁旧标签
        if circle_label is not None:
            circle_label.destroy()
        if circle_text_label is not None:
            circle_text_label.destroy()

        # 创建新标签
        circle_text_label = Label(circle_frame, text=result_text, font=("Arial", 10, "bold"), fg="#0066cc")
        circle_text_label.pack(pady=10)

        update_status(f"✅ 圆形检测完成：{circle_count}个圆形 | 详情见Excel")
    except Exception as e:
        logging.error(f"圆形检测失败：{str(e)}")
        update_status(f"❌ 圆形检测失败：{str(e)} | 请检查分割结果")


def update_status(msg):
    """更新底部状态提示"""
    global status_label
    if status_label is not None:
        status_label.config(text=msg, wraplength=900, font=("Arial", 10), fg="#666")


# -------------------------- 4. 主程序入口 --------------------------
if __name__ == "__main__":
    # 创建主窗口
    root = Tk()
    root.title("U-Net 图像分割与圆形检测工具")
    root.geometry("1200x800")
    root.resizable(width=False, height=False)

    # 初始化Tkinter变量
    binary_flag = IntVar(value=0)
    gaussian_flag = IntVar(value=0)

    # 顶部控制区
    control_frame = Frame(root, relief="groove", bd=1)
    control_frame.pack(pady=15, fill="x", padx=20)

    Button(control_frame, text="1. 加载模型权重", command=load_model, width=16, bg="#e6f7ff").pack(side="left", padx=8,
                                                                                             pady=10)
    Button(control_frame, text="2. 选择原始图像", command=select_image, width=16, bg="#e6f7ff").pack(side="left", padx=8,
                                                                                               pady=10)

    preprocess_ctrl = Frame(control_frame)
    preprocess_ctrl.pack(side="left", padx=8, pady=10)
    Checkbutton(preprocess_ctrl, text="二值化", variable=binary_flag, font=("Arial", 9)).pack(side="left", padx=4)
    Checkbutton(preprocess_ctrl, text="高斯去噪", variable=gaussian_flag, font=("Arial", 9)).pack(side="left", padx=4)
    Button(preprocess_ctrl, text="3. 应用预处理", command=apply_preprocessing, width=14, bg="#e6f7ff").pack(side="left",
                                                                                                       padx=4)

    Button(control_frame, text="4. 执行分割", command=start_analysis, width=14, bg="#e6f7ff").pack(side="left", padx=8,
                                                                                               pady=10)
    Button(control_frame, text="5. 检测圆形", command=detect_circles_and_save_count, width=14, bg="#fff2cc").pack(
        side="left", padx=8, pady=10)

    # 中间预览区
    preview_main = Frame(root)
    preview_main.pack(pady=10, padx=20)

    original_frame = Frame(preview_main, width=220, height=320, bd=2, relief="groove")
    original_frame.pack(side="left", padx=15, pady=5)
    original_frame.pack_propagate(False)

    preprocess_frame = Frame(preview_main, width=220, height=320, bd=2, relief="groove")
    preprocess_frame.pack(side="left", padx=15, pady=5)
    preprocess_frame.pack_propagate(False)

    segment_frame = Frame(preview_main, width=220, height=320, bd=2, relief="groove")
    segment_frame.pack(side="left", padx=15, pady=5)
    segment_frame.pack_propagate(False)

    overlay_frame = Frame(preview_main, width=220, height=320, bd=2, relief="groove")
    overlay_frame.pack(side="left", padx=15, pady=5)
    overlay_frame.pack_propagate(False)

    circle_frame = Frame(preview_main, width=220, height=320, bd=2, relief="groove")
    circle_frame.pack(side="left", padx=15, pady=5)
    circle_frame.pack_propagate(False)

    # 底部状态区
    status_frame = Frame(root, relief="sunken", bd=1)
    status_frame.pack(pady=15, fill="x", padx=20)
    status_label = Label(
        status_frame,
        text="请先加载模型（步骤1） | 操作流程：1.加载模型→2.选图→3.预处理→4.分割→5.检测圆形",
        font=("Arial", 10),
        fg="#666",
        anchor="w",
        padx=10,
        pady=5
    )
    status_label.pack(fill="x")

    # 启动主循环
    root.mainloop()