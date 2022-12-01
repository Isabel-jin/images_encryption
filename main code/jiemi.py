import numpy as np

def jiemi(x,data):
    # m0 m1 m2 为三个混沌序列，用密钥赋初值
    height, width = x.shape
    size = height * width

    m0 = np.array(np.zeros(size))
    m1 = np.array(np.zeros(size))
    m2 = np.array(np.zeros(size))
    m0[0] = data[0]
    m1[0] = data[1]
    m2[0] = data[2]

    # u0 u1 u2 为三个混沌序列参数，用密钥赋值
    u0 = data[3]

    for i in range(1,size):
        m0[i] =  u0 * m0[i-1] * (1 - m0[i-1])
    #print(m0)
    m0 = 255 * m0 % 256
    m0 = m0.astype(np.int32)

    u1 = data[4]

    for i in range(1, size):
        m1[i] = u1 * m1[i-1] * (1 - m1[i-1])

    m1 = 255 * m1 % 256
    m1 = m1.astype(np.int32)

    u2 = data[5]

    for i in range(1, size):
        m2[i] = u2 * m2[i-1] * (1 - m2[i-1])

    m2 = 255 * m2 % 256
    m2 = m2.astype(np.int32)

    sigma=data[6]

    n = 0

    c = np.zeros((height, width),np.int32)
    for i in range(height):
        for j in range(width):
           c[i,j] = m0[n] ^ m1[n]
           c[i,j] = c[i,j]^m2[n]
           c[i,j] = x[i,j] - c[i,j]
           c[i,j] = c[i,j] % 256
           n = n+1
    c = c.astype(np.uint8)

    return c

