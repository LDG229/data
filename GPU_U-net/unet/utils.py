import torch
import torch.nn.functional as F


def binarize_mask(x, threshold=0.5):
    """
    将预测图或人工标注图转换为二值边界图。
    适用于红色边界图、白色边界图、三通道输出图。

    输入:
        x: Tensor，形状为 [B, C, H, W] 或 [C, H, W]
    输出:
        binary: Tensor，形状为 [B, 1, H, W]
    """

    # 如果输入是 [C, H, W]，增加 batch 维度
    if x.dim() == 3:
        x = x.unsqueeze(0)

    # 如果是三通道图像，不取平均，而是取三个通道中的最大值
    # 原因：你的标注边界是红色，RGB大约为[1,0,0]
    # 如果取平均会变成0.333，低于0.5阈值，导致边界被误判为背景
    if x.shape[1] == 3:
        x = torch.max(x, dim=1, keepdim=True)[0]

    binary = (x > threshold).float()

    return binary


def dilate_boundary(boundary, tolerance=2):
    """
    对边界进行膨胀，允许一定范围内的边界偏移。
    tolerance=2 表示允许约2个像素的边界误差。
    """

    kernel_size = 2 * tolerance + 1

    dilated = F.max_pool2d(
        boundary,
        kernel_size=kernel_size,
        stride=1,
        padding=tolerance
    )

    return dilated


def boundary_f1_score(pred, target, threshold=0.5, tolerance=2, eps=1e-7):
    """
    计算 Boundary F1-score。

    pred: 模型预测结果
    target: 人工标注边界图
    threshold: 二值化阈值
    tolerance: 边界容忍范围
    """

    pred_b = binarize_mask(pred, threshold)
    target_b = binarize_mask(target, threshold)

    pred_dilated = dilate_boundary(pred_b, tolerance)
    target_dilated = dilate_boundary(target_b, tolerance)

    # 边界精确率：预测边界中有多少落在真实边界附近
    boundary_precision = (pred_b * target_dilated).sum(dim=(1, 2, 3)) / (
        pred_b.sum(dim=(1, 2, 3)) + eps
    )

    # 边界召回率：真实边界中有多少被预测边界覆盖
    boundary_recall = (target_b * pred_dilated).sum(dim=(1, 2, 3)) / (
        target_b.sum(dim=(1, 2, 3)) + eps
    )

    boundary_f1 = 2 * boundary_precision * boundary_recall / (
        boundary_precision + boundary_recall + eps
    )

    return (
        boundary_f1.mean().item(),
        boundary_precision.mean().item(),
        boundary_recall.mean().item()
    )


def boundary_iou(pred, target, threshold=0.5, tolerance=2, eps=1e-7):
    """
    计算 Boundary IoU。
    只比较边界附近区域的交并比。
    """

    pred_b = binarize_mask(pred, threshold)
    target_b = binarize_mask(target, threshold)

    pred_band = dilate_boundary(pred_b, tolerance)
    target_band = dilate_boundary(target_b, tolerance)

    intersection = (pred_band * target_band).sum(dim=(1, 2, 3))
    union = ((pred_band + target_band) > 0).float().sum(dim=(1, 2, 3))

    biou = intersection / (union + eps)

    return biou.mean().item()
