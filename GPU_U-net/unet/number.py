import cv2
import numpy as np
import pandas as pd

def detect_circles(image_path):
    # 读取彩色图像
    img = cv2.imread(image_path)

    # 检查图像是否成功加载
    if img is None:
        print("错误：无法读取图像，请检查文件路径！")
        return 0

    # 图像预处理
    # 中值滤波去除椒盐噪声
    median = cv2.medianBlur(img, 5)
    # 高斯模糊减少噪声
    blurred = cv2.GaussianBlur(median, (5, 5), 0)
    # 转换为灰度图像
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # 形态学操作：开运算去除小的噪声点，闭运算填充圆内部的空洞
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # 霍夫圆检测
    circles = cv2.HoughCircles(closing, cv2.HOUGH_GRADIENT, 1,
                               minDist=30,  # 圆与圆之间的最小距离
                               param1=50,   # Canny边缘检测的高阈值
                               param2=10,   # 累加器阈值，值越小，检测到的圆越多
                               minRadius=10,  # 最小圆半径
                               maxRadius=100) # 最大圆半径

    # 验证检测到的圆
    valid_circles = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # 计算圆的面积
            area = np.pi * r * r
            # 过滤面积过小或过大的圆
            if 100 < area < 10000:
                # 计算圆度
                perimeter = 2 * np.pi * r
                contour = np.array([[[x + r, y]], [[x - r, y]], [[x, y + r]], [[x, y - r]]], dtype=np.int32)
                _, radius = cv2.minEnclosingCircle(contour)
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                # 过滤圆度不符合要求的圆
                if circularity > 0.8:
                    valid_circles.append((x, y, r))

    # 统计有效圆的数量
    circle_count = len(valid_circles)
    print(f"\n检测到的有效圆数量：{circle_count} 个")

    # 在原图上绘制结果
    result = img.copy()
    for (x, y, r) in valid_circles:
        cv2.circle(result, (x, y), r, (0, 255, 0), 2)
        cv2.rectangle(result, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    # 显示计数结果
    cv2.putText(result, f"Count: {circle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示结果图像
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 将圆的信息保存到 Excel 文件中
    circle_info = []
    for (x, y, r) in valid_circles:
        radius = r
        diameter = 2 * r
        perimeter = 2 * np.pi * r
        area = np.pi * r * r
        circle_info.append([radius, diameter, perimeter, area])

    df = pd.DataFrame(circle_info, columns=['半径', '直径', '周长', '面积'])
    df.to_excel('result/circle_data.xlsx', index=False)

    return circle_count

if __name__ == "__main__":
    image_path = r'E:\GPU_U-net\image_use\mask\1.jpg'
    detect_circles(image_path)