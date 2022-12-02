import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from jiami import jiami
from jiemi import jiemi
from evalution import evaluate
from evalution import gauss_noise
from evalution import salt_and_pepper_noise
from evalution import occlusion
# 密钥 0.343 0.432 0.63 3.769 3.82 3.8 0.1 1


if __name__ == '__main__':
    x = cv.imread('lenaRGB.bmp', cv.IMREAD_COLOR)
    print(x.shape)
    x0 = x[:, :, 0]
    x1 = x[:, :, 1]
    x2 = x[:, :, 2]
    r = np.array(np.zeros(8))
    print(r.dtype)
    print("请输入加密密钥：")
    arr = input("")
    r = [float(n) for n in arr.split()]
    print(r)
    e0 = jiami(x0, r)
    e1 = jiami(x1, r)
    e2 = jiami(x2, r)
    e = np.dstack((e0, e1, e2))
    #e1 e2 e3 为加密三个通道#

    c0 = jiemi(e0, r)
    c1 = jiemi(e1, r)
    c2 = jiemi(e2, r)
    c = np.dstack((c0, c1, c2))
    #c1 c2 c3 为解密的三个通道#

    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.imshow(cv.cvtColor(x, cv.COLOR_BGR2RGB))
    plt.title('origin')
    plt.subplot(222)
    plt.imshow(cv.cvtColor(e, cv.COLOR_BGR2RGB))
    plt.title('jiami')
    plt.subplot(223)
    plt.imshow(cv.cvtColor(c, cv.COLOR_BGR2RGB))
    plt.title('jiemi')
    plt.savefig("result.png")
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.hist(x.ravel(), 256, [0, 255])
    plt.title('hist_origin')
    plt.subplot(122)
    plt.hist(e.ravel(), 256, [0, 255])
    plt.title('hist_e')
    plt.show()

    #阻塞攻击
    e_occlusion = occlusion(e)
    # 添加噪声测试鲁棒性
    # 获取噪声图像
    e_gauss = gauss_noise(e)
    e_salt_and_pepper = salt_and_pepper_noise(e)
    # 拆分为三个通道
    e_gauss_0 = e_gauss[:, :, 0]
    e_gauss_1 = e_gauss[:, :, 1]
    e_gauss_2 = e_gauss[:, :, 2]
    e_salt_and_pepper_0 = e_salt_and_pepper[:, :, 0]
    e_salt_and_pepper_1 = e_salt_and_pepper[:, :, 1]
    e_salt_and_pepper_2 = e_salt_and_pepper[:, :, 2]
    e_occlusion_0 = e_occlusion[:, :, 0]
    e_occlusion_1 = e_occlusion[:, :, 1]
    e_occlusion_2 = e_occlusion[:, :, 2]
    # 解密
    c_gauss_0 = jiemi(e_gauss_0, r)
    c_gauss_1 = jiemi(e_gauss_1, r)
    c_gauss_2 = jiemi(e_gauss_2, r)
    c_gauss = np.dstack((c_gauss_0, c_gauss_1, c_gauss_2))
    c_salt_and_pepper_0 = jiemi(e_salt_and_pepper_0, r)
    c_salt_and_pepper_1 = jiemi(e_salt_and_pepper_1, r)
    c_salt_and_pepper_2 = jiemi(e_salt_and_pepper_2, r)
    c_salt_and_pepper = np.dstack((c_salt_and_pepper_0, c_salt_and_pepper_1, c_salt_and_pepper_2))
    c_occlusion_0 = jiemi(e_occlusion_0, r)
    c_occlusion_1 = jiemi(e_occlusion_1, r)
    c_occlusion_2 = jiemi(e_occlusion_2, r)
    c_occlusion = np.dstack((c_occlusion_0, c_occlusion_1, c_occlusion_2))
    # 结果展示
    plt.figure(figsize=(8, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.subplot(221)
    plt.imshow(cv.cvtColor(e_gauss, cv.COLOR_BGR2RGB))
    plt.title('高斯噪声图像')
    plt.subplot(222)
    plt.imshow(cv.cvtColor(c_gauss, cv.COLOR_BGR2RGB))
    plt.title('解密后图像')
    plt.subplot(223)
    plt.imshow(cv.cvtColor(e_salt_and_pepper, cv.COLOR_BGR2RGB))
    plt.title('椒盐噪声图像')
    plt.subplot(224)
    plt.imshow(cv.cvtColor(c_salt_and_pepper, cv.COLOR_BGR2RGB))
    plt.title('解密后图像')
    plt.show()


    plt.subplot(121)
    plt.imshow(cv.cvtColor(e_occlusion, cv.COLOR_BGR2RGB))
    plt.title('阻塞攻击后的加密图像')
    plt.subplot(122)
    plt.imshow(cv.cvtColor(c_occlusion, cv.COLOR_BGR2RGB))
    plt.title('解密后图像')
    plt.show()

    #对加密算法的安全性进行评估
    evaluate(x, e)
