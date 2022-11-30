import numpy as np
import cv2 as cv
# 坐标 20 20 40 40


def reyinxie(x):
    '''
    :param x: 准备被解隐写的图片
    :return data: 隐写后得到的data
    '''
    # x1为隐写后的图像的R通道转化成一维
    x1 = x[:, :, 0].ravel()

    # 首先考虑提取前16个像素，读出box个数
    # y0存储前16像素的R通道坐标
    y0 = np.zeros(16, np.uint8)
    # y2存储16个像素R通道的16进制str形式
    y2 = []
    # y3存储len*16个最低位
    y3 = np.zeros(16)

    for i in range(16):
        y0[i] = x1[i]
        y2.append(format(int(y0[i]), 'b').zfill(8))
        #print(y0[i])
        # list1对应每个像素
        list1 = list(y2[i])
        #print(list1)
        y3[i] = list1[7]

    num = 0
    for w in range(15):
        num = num + pow(y3[w] * 2, 15 - w)
    num = num + y3[w + 1]

    # 从第一个读到的数得到整个data的长度
    lenner = int(4 * num + 1)
    print(num)

    # x0存储前len*16像素的R通道坐标
    x0 = np.zeros(lenner * 16, np.uint8)
    # x2存储len*16个像素R通道的16进制str形式
    x2 = []
    # x3存储len*16个最低位
    x3 = np.zeros(lenner * 16)

    for i in range(lenner * 16):
        x0[i] = x1[i]
        x2.append(format(int(x0[i]), 'b').zfill(8))
        #print(x0[i])
        # list1对应每个像素
        list1 = list(x2[i])
        #print(list1)
        x3[i] = list1[7]

    x3 = x3.reshape(-1, 16)
    # out用于存储解隐写得到的各个坐标作为输出
    out = np.zeros(lenner)
    for k in range(lenner):
        for w in range(15):
            out[k] = out[k] + pow(x3[k, w] * 2, 15 - w)
        out[k] = out[k] + x3[k, w + 1]

    out = out.astype(int)
    return out


if __name__ == '__main__':
    x = cv.imread('./result/yinxie.bmp', cv.IMREAD_COLOR)
    data = reyinxie(x)
    print(data)