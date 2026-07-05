import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取两幅图像
img1 = cv2.imread('C:/Users/27274/Desktop/photo1.jpg', cv2.IMREAD_GRAYSCALE)  # 第一幅图像
img2 = cv2.imread('C:/Users/27274/Desktop/photo2.jpg', cv2.IMREAD_GRAYSCALE)  # 第二幅图像

# 检查图像是否成功读取
if img1 is None:
    raise ValueError("无法读取第一幅图像，请检查路径。")
if img2 is None:
    raise ValueError("无法读取第二幅图像，请检查路径。")

# 初始化 SIFT 检测器
sift = cv2.SIFT_create()

# 检测关键点和计算描述符
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 使用 Brute-Force 匹配器
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# 匹配描述符
matches = bf.match(des1, des2)

# 按距离排序
matches = sorted(matches, key=lambda x: x.distance)

# 选择前 50 个最佳匹配
good_matches = matches[:50]

# 提取匹配点的坐标
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 绘制匹配结果
match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)

# 显示匹配结果
plt.figure(figsize=(12, 6))
plt.imshow(match_img, cmap='gray')
plt.title('SIFT Feature Matching')
plt.show()

# 可选：使用 RANSAC 进行几何验证并计算单应性矩阵
MIN_MATCH_COUNT = 10
if len(good_matches) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    match_img_ransac = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)

    plt.figure(figsize=(12, 6))
    plt.imshow(match_img_ransac, cmap='gray')
    plt.title('SIFT Feature Matching with RANSAC')
    plt.show()
else:
    print("Not enough matches are found - %d/%d" % (len(good_matches), MIN_MATCH_COUNT))