import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

#计算像素数变化率

def NPCR(img1,img2):
  #opencv颜色通道顺序为BGR
  w,h,_=img1.shape

  #图像通道拆分
  B1,G1,R1=cv2.split(img1)
  B2,G2,R2=cv2.split(img2)

  #返回数组的排序后的唯一元素和每个元素重复的次数
  ar,num=np.unique((R1!=R2),return_counts=True)
  R_npcr=(num[0] if ar[0]==True else num[1])/(w*h)
  ar,num=np.unique((G1!=G2),return_counts=True)
  G_npcr=(num[0] if ar[0]==True else num[1])/(w*h)
  ar,num=np.unique((B1!=B2),return_counts=True)
  B_npcr=(num[0] if ar[0]==True else num[1])/(w*h)

  return R_npcr,G_npcr,B_npcr


#两张图像之间的平均变化强度

def UACI(img1,img2):
  w,h,_=img1.shape
  #图像通道拆分
  B1,G1,R1=cv2.split(img1)
  B2,G2,R2=cv2.split(img2)
  #元素为uint8类型取值范围：0到255

  # print(R1.dtype)

  #强制转换元素类型，为了运算
  R1=R1.astype(np.int16)
  R2=R2.astype(np.int16)
  G1=G1.astype(np.int16)
  G2=G2.astype(np.int16)
  B1=B1.astype(np.int16)
  B2=B2.astype(np.int16)

  sumR=np.sum(abs(R1-R2))
  sumG=np.sum(abs(G1-G2))
  sumB=np.sum(abs(B1-B2))
  R_uaci=sumR/255/(w*h)
  G_uaci=sumG/255/(w*h)
  B_uaci=sumB/255/(w*h)

  return R_uaci,G_uaci,B_uaci

def RBG_correlation(channel, N):
    hight, width = channel.shape
    row = np.random.randint(0, hight-1, N)
    col = np.random.randint(0, width-1, N)
    x = []
    y = []
    #垂直相邻像素
    v_y = []
    #水平相邻像素
    h_y = []
    #对角线相邻像素
    d_y = []
    for i in range(N):
        x.append(channel[row[i]][col[i]])
        h_y.append(channel[row[i]][col[i]+1])
        v_y.append(channel[row[i]+1][col[i]])
        d_y.append(channel[row[i]+1][col[i]+1])

    #三个方向合在一起
    x = x*3
    y = h_y + v_y + d_y

    #求Ex
    ex = 0
    for i in range(N):
        ex = ex + channel[row[i]][col[i]]
    ex = ex / N

    #求Dx
    dx = 0
    for i in range(N):
        dx = dx + (channel[row[i]][col[i]]-ex)**2
    dx = dx / N

    # 求Ey
    h_ey = 0
    v_ey = 0
    d_ey = 0
    for i in range(N):
        h_ey = h_ey + channel[row[i]][col[i] + 1]
        v_ey = v_ey + channel[row[i] + 1][col[i]]
        d_ey = d_ey + channel[row[i] + 1][col[i] + 1]
    h_ey = h_ey / N
    v_ey = v_ey / N
    d_ey = d_ey / N

    # 求Dy
    h_dy = 0
    v_dy = 0
    d_dy = 0
    for i in range(N):
        h_dy = h_dy + (channel[row[i]][col[i] + 1] - h_ey) ** 2
        v_dy = v_dy + (channel[row[i] + 1][col[i]] - v_ey) ** 2
        d_dy = d_dy + (channel[row[i] + 1][col[i] + 1] - d_ey) ** 2
    h_dy = h_dy / N
    v_dy = v_dy / N
    d_dy = d_dy / N

    #求协方差
    h_cov = 0
    v_cov = 0
    d_cov = 0
    for i in range(N):
        h_cov = h_cov + (channel[row[i]][col[i]] - ex) * (channel[row[i]][col[i] + 1] - h_ey)
        v_cov = v_cov + (channel[row[i]][col[i]] - ex) * (channel[row[i] + 1][col[i]] - v_ey)
        d_cov = d_cov + (channel[row[i]][col[i]] - ex) * (channel[row[i] + 1][col[i] + 1] - d_ey)
    v_cov = v_cov / N
    h_cov = h_cov / N
    d_cov = d_cov / N
    h_Rxy = h_cov / (np.sqrt(dx) * np.sqrt(h_dy))
    v_Rxy = v_cov / (np.sqrt(dx) * np.sqrt(v_dy))
    d_Rxy = d_cov / (np.sqrt(dx) * np.sqrt(d_dy))
    return h_Rxy, v_Rxy, d_Rxy, x, y

def correlation(img, N=3000):
    hight, weight, _ = img.shape
    B, G, R = cv2.split(img)
    R_Rxy = RBG_correlation(R, N)
    B_Rxy = RBG_correlation(B, N)
    G_Rxy = RBG_correlation(G, N)
    return R_Rxy, B_Rxy, G_Rxy

#对解密图像添加均值为0，方差为0.001的高斯噪声
def gauss_noise(img, mean = 0, var = 0.001):
    img = np.array(img / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    out = img + noise
    if out.min() < 0:
        low_clip = -1
    else:
        low_clip = 0
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out

#添加默认为10%的椒盐噪声
def salt_and_pepper_noise(img, proportation = 0.1):
    height, width, _ = img.shape
    num = int(height * width * proportation)
    for i in range(num):
        w = random.randint(0, width - 1)
        h = random.randint(0, height - 1)
        if random.randint(0, 1) == 0:
            img[h, w] = 0
        else:
            img[h,w] = 255
    return img

#阻塞攻击
def occlusion(img):
    height, width, _ = img.shape
    B, G, R = cv2.split(img)
    #随机移除R通道中80*80大小的像素块
    #产生随机整数
    R_w = random.randint(0, width - 80)
    R_h = random.randint(0, height - 80)
    for i in range(80):
        for j in range(80):
            R[R_h + i][R_w + j] = 0
    #随机移除G通道中50*80大小的像素块
    #产生随机整数
    G_w = random.randint(0, width - 50)
    G_h = random.randint(0, height - 80)
    for i in range(80):
        for j in range(50):
            G[G_h + i][G_w + j] = 0
    #随机移除B通道中80*80大小的像素块
    #产生随机整数
    #B_w = random.randint(0, width - 80)
    #B_h = random.randint(0, height - 80)
    #for i in range(80):
    #    for j in range(80):
    #        B[B_h + i][B_w + j] = 0
    out = cv2.merge([R, G, B])
    #随机移除全通道中60*50的像素块
    a_w = random.randint(0, width - 60)
    a_h = random.randint(0, height - 50)
    for i in range(50):
        for j in range(60):
            out[a_h + i][a_w + j] = np.array([0, 0, 0])
    return out

def evaluate(x,e):
    #x为原图像，e为加密后的图像
    # 图像相关性
    R1_Rxy, B1_Rxy, G1_Rxy = correlation(x)
    R2_Rxy, B2_Rxy, G2_Rxy = correlation(e)
    # 结果展示
    # 明文图像相邻像素的相关性接近 1，而密文图像相邻像素的相关性应该接近于 0
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.subplot(221)
    plt.imshow(x[:, :, (2, 1, 0)])
    plt.title('原图像')
    plt.subplot(222)
    plt.scatter(R1_Rxy[3], R1_Rxy[4], s=1, c='red')
    plt.title('通道R')
    plt.subplot(223)
    plt.scatter(G1_Rxy[3], G1_Rxy[4], s=1, c='green')
    plt.title('通道G')
    plt.subplot(224)
    plt.scatter(B1_Rxy[3], B1_Rxy[4], s=1, c='blue')
    plt.title('通道B')
    plt.show()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.subplot(221)
    plt.imshow(e[:, :, (2, 1, 0)])
    plt.title('加密后图像')
    plt.subplot(222)
    plt.scatter(R2_Rxy[3], R2_Rxy[4], s=1, c='red')
    plt.title('通道R')
    plt.subplot(223)
    plt.scatter(G2_Rxy[3], G2_Rxy[4], s=1, c='green')
    plt.title('通道G')
    plt.subplot(224)
    plt.scatter(B2_Rxy[3], B2_Rxy[4], s=1, c='blue')
    plt.title('通道B')
    plt.show()

    r_npcr,g_npcr,b_npcr = NPCR(x, e)
    r_uaci,g_uaci,b_uaci = UACI(x, e)
    #它们相应的理想值分别为
    #NPCR = 99.6094％，UACI = 33.4635 %

    print('*****NPCR*****')
    print('RED   :{:.4%}'.format(r_npcr))
    print('GREEN :{:.4%}'.format(g_npcr))
    print('BLUE  :{:.4%}'.format(b_npcr))

    print('*****UACI*****')
    print('RED   :{:.4%}'.format(r_uaci))
    print('GREEN :{:.4%}'.format(g_uaci))
    print('BLUE  :{:.4%}'.format(b_uaci))

