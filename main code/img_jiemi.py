from jiemi import jiemi
from reyinxie import reyinxie
import copy
import numpy as np
def img_jiemi(img , arr):  #传入Img和密码
    image=copy.deepcopy(img)
    yinxie_data = reyinxie(img) #解隐写data
    print("jie yinxie",yinxie_data)
    for i in range(yinxie_data[0]):
        imgcut = img[yinxie_data[4*i+2]:yinxie_data[4*i+4],yinxie_data[4*i+1]:yinxie_data[4*i+3]]
        cut0=imgcut[ :, :,0]
        cut1=imgcut[ :, :,1]
        cut2=imgcut[ :, :,2]
        newcut0 = jiemi(cut0,arr)
        newcut1 = jiemi(cut1,arr)
        newcut2 = jiemi(cut2,arr)
        newcut = np.dstack((newcut0,newcut1,newcut2))
        image[yinxie_data[4*i+2]:yinxie_data[4*i+4],yinxie_data[4*i+1]:yinxie_data[4*i+3]] = newcut
    return image
        
