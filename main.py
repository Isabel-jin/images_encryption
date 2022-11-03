import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from jiami import jiami
from jiemi import jiemi
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

    c0 = jiemi(e0, r)
    c1 = jiemi(e1, r)
    c2 = jiemi(e2, r)
    c = np.dstack((c0, c1, c2))

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





