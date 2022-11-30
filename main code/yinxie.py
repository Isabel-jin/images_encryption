import numpy as np
import cv2 as cv
# 坐标 20 20 40 40


def yinxie(img, arr):
    '''
    :param img: 待隐写入数据的图片
    :param arr: 待被隐写的数据
    :return x: 隐写后的图片
    '''
    x = np.copy(img)
    #len存储待隐写整数的个数
    lenner = len(arr)
    r = np.array(np.zeros(lenner))
    # num存储四个str，为len个坐标值的十六进制表示
    num = []
    for i in range(lenner):
        r[i] = float(arr[i])
        num.append(format(arr[i], 'b').zfill(16))
        print(num[i])

    # 对原图像的前16*len个像素的R通道进行隐写
    # X0为图像的R通道转换成一维
    x0 = x[:, :, 0].ravel()
    # x1存储前16*len个像素点的16进制形式str
    x1 = []
    # x2存储len*16个uint8值
    x2 = np.zeros(lenner*16, np.uint8)
    for i in range(lenner):
        for j in range(16):
            # k为待隐写的像素标号
            k = 16 * i + j
            x1.append(format(int(x0[k]), 'b').zfill(8))
            # list1是16进制数字
            list1 = list(num[i])
            # list2是每个像素
            list2 = list(x1[k])
            # 将像素最低位修改成待隐写的值
            list2[7] = list1[j]
            #转换成十进制
            for w in range(7):
                x2[k] = x2[k] + pow(int(list2[w]) * 2, 7-w)
            x2[k] = x2[k] + int(list2[7])
            # 改变原图像的前16*len个像素
            x0[k] = x2[k]
    x0 = x0.reshape(x[:, :, 0].shape)
    x[:, :, 0] = x0
    # 必须用bmp格式存储，保证没有压缩
    return x


if __name__ == '__main__':
    img = cv.imread('lenaRGB.bmp', cv.IMREAD_COLOR)
    #待隐写的数据data为一维数组，存储各个box的坐标[num,x11,y11,x12,y12,x21,y21,x22,y22...]
    #第一个位置存放矩形个数，便于解隐写
    data = [2, 21, 20, 40, 40, 60, 60, 81, 80]
    x = yinxie(img, data)
    cv.imwrite('./result/yinxie.bmp', x)