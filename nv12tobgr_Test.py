
import numpy as np
import cv2


def bgr2nv12_opencv(image):
    height, width = image.shape[0], image.shape[1]
    area = height * width
    yuv420p = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420).reshape((area * 3 // 2,))
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

    nv12 = np.zeros_like(yuv420p)
    nv12[:height * width] = y
    nv12[height * width:] = uv_packed
    return nv12



if __name__ == '__main__':
    img_file = cv2.imread('./kite.jpg')
    height, width = img_file.shape[0], img_file.shape[1]
    nv12_data = bgr2nv12_opencv(img_file)
    bgr_img = cv2.cvtColor(nv12_data.reshape((height*3//2, width)), cv2.COLOR_YUV2BGR_NV12)   # nv12转BGR
    cv2.imwrite("newimg.jpg",bgr_img)