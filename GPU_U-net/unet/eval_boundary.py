import os
import csv
import torch
from torch.utils.data import DataLoader

from data import MyDataset
from net import UNet
from utils import boundary_f1_score, boundary_iou, binarize_mask


# ==========================
# 1. 基本路径设置
# ==========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型权重路径
weight_path = '../params/model_weights.pth'

# 数据集路径
# 该路径下必须包含 image 和 mask 两个文件夹
data_path = r'E:\桌面\GPU_U-net\image_use'

# 结果保存路径
result_csv_path = 'result/boundary_metrics.csv'


# ==========================
# 2. 评价函数
# ==========================

def evaluate_boundary_metrics():
    print("当前设备:", device)

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"模型权重文件不存在：{weight_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集路径不存在：{data_path}")

    if not os.path.exists(os.path.join(data_path, 'image')):
        raise FileNotFoundError(f"image 文件夹不存在：{os.path.join(data_path, 'image')}")

    if not os.path.exists(os.path.join(data_path, 'mask')):
        raise FileNotFoundError(f"mask 文件夹不存在：{os.path.join(data_path, 'mask')}")

    os.makedirs('result', exist_ok=True)

    # 加载数据集
    dataset = MyDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 加载模型
    net = UNet().to(device)
    net.load_state_dict(torch.load(weight_path, map_location=device))
    net.eval()

    print("模型加载成功，开始计算 Boundary F1-score 和 Boundary IoU...\n")

    bf1_list = []
    biou_list = []
    bp_list = []
    br_list = []

    csv_rows = []

    with torch.no_grad():
        for i, (image, mask) in enumerate(dataloader):
            image = image.to(device)
            mask = mask.to(device)

            # 模型预测
            pred = net(image)

            # 计算边界指标
            bf1, bp, br = boundary_f1_score(
                pred,
                mask,
                threshold=0.5,
                tolerance=2
            )

            biou = boundary_iou(
                pred,
                mask,
                threshold=0.5,
                tolerance=2
            )

            # 检查预测边界像素数和真实边界像素数
            pred_b = binarize_mask(pred, threshold=0.5)
            mask_b = binarize_mask(mask, threshold=0.5)

            pred_pixel_count = pred_b.sum().item()
            mask_pixel_count = mask_b.sum().item()

            bf1_list.append(bf1)
            biou_list.append(biou)
            bp_list.append(bp)
            br_list.append(br)

            csv_rows.append([
                i + 1,
                bf1,
                biou,
                bp,
                br,
                pred_pixel_count,
                mask_pixel_count
            ])

            print(
                f"第 {i + 1} 张图："
                f"Boundary F1 = {bf1:.4f}, "
                f"Boundary IoU = {biou:.4f}, "
                f"Boundary Precision = {bp:.4f}, "
                f"Boundary Recall = {br:.4f}, "
                f"预测边界像素数 = {pred_pixel_count:.0f}, "
                f"真实边界像素数 = {mask_pixel_count:.0f}"
            )

    # ==========================
    # 3. 计算平均值
    # ==========================

    mean_bf1 = sum(bf1_list) / len(bf1_list)
    mean_biou = sum(biou_list) / len(biou_list)
    mean_bp = sum(bp_list) / len(bp_list)
    mean_br = sum(br_list) / len(br_list)

    print("\n========== 边界评价结果 ==========")
    print(f"Boundary Precision: {mean_bp:.4f}")
    print(f"Boundary Recall:    {mean_br:.4f}")
    print(f"Boundary F1-score:  {mean_bf1:.4f}")
    print(f"Boundary IoU:       {mean_biou:.4f}")
    print("==================================")



    # ==========================
    # 4. 保存到 CSV 文件
    # ==========================

    with open(result_csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)

        writer.writerow([
            'image_index',
            'Boundary_F1',
            'Boundary_IoU',
            'Boundary_Precision',
            'Boundary_Recall',
            'Pred_Boundary_Pixels',
            'GT_Boundary_Pixels'
        ])

        writer.writerows(csv_rows)

        writer.writerow([])
        writer.writerow(['Mean Boundary F1', mean_bf1])
        writer.writerow(['Mean Boundary IoU', mean_biou])
        writer.writerow(['Mean Boundary Precision', mean_bp])
        writer.writerow(['Mean Boundary Recall', mean_br])

    print(f"\n每张图的边界评价结果已保存至：{result_csv_path}")


if __name__ == '__main__':
    evaluate_boundary_metrics()
