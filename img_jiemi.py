from jiemi import jiemi
from reyinxie import reyinxie
import numpy as np
import copy

def img_jiemi(img , arr):  #传入Img和密码
    image=copy.deepcopy(img)
    yinxie_data = reyinxie(img) #解隐写data
    for i in range(yinxie_data[0]):
        imgcut = img[4*i+2:4*i+4,4*i+1:4*i+3]
        newcut = jiemi(imgcut,arr)
        image[4*i+2:4*i+4,4*i+1:4*i+3] = newcut
    return image
        
