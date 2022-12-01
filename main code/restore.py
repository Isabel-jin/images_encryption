import numpy as np
import cv2
import postprocess as pc
from jiami import jiami
import colorsys
from yinxie import yinxie


def predict_cut(model_output,
                model_hw_shape,
                origin_image=None,
                origin_img_shape=None,
                score_threshold=0.30,
                nms_threshold=0.45,
                dump_image=True):
    num_classes = 80
    anchors = np.array([
        1.25, 1.625, 2.0, 3.75, 4.125, 2.875, 1.875, 3.8125, 3.875, 2.8125,
        3.6875, 7.4375, 3.625, 2.8125, 4.875, 6.1875, 11.65625, 10.1875
    ]).reshape((3, 3, 2)) #转换成三维数组
    num_anchors = anchors.shape[0] #那么这里应该等于3了
    strides = np.array([8, 16, 32])
    input_shape = (416, 416)

    if origin_image is not None:
        org_height, org_width = origin_image.shape[0:2]
    else:
        org_height, org_width = origin_img_shape
    process_height, process_width = model_hw_shape

    pred_sbbox = model_output[2].buffer.transpose([0, 3, 1, 2])

    pred_mbbox = model_output[1].buffer.transpose([0, 3, 1, 2])

    pred_lbbox = model_output[0].buffer.transpose([0, 3, 1, 2])

    pred_sbbox = pc.yolo_decoder(pred_sbbox, num_anchors, num_classes, anchors[0],
                              strides[0])
    pred_mbbox = pc.yolo_decoder(pred_mbbox, num_anchors, num_classes, anchors[1],
                              strides[1])
    pred_lbbox = pc.yolo_decoder(pred_lbbox, num_anchors, num_classes, anchors[2],
                              strides[2])

    pred_bbox = np.concatenate([
        np.reshape(pred_sbbox, (-1, 5 + num_classes)),
        np.reshape(pred_mbbox, (-1, 5 + num_classes)),
        np.reshape(pred_lbbox, (-1, 5 + num_classes))
    ],
                               axis=0)

    bboxes = pc.postprocess_boxes(pred_bbox, (org_height, org_width),
                               input_shape=(process_height, process_width),
                               score_threshold=score_threshold)
    nms_bboxes = pc.nms(bboxes, nms_threshold) ##找到best box
    if dump_image and origin_image is not None:
        print("detected item num: ", len(nms_bboxes))
        new_img=imgRestore(origin_image, nms_bboxes)
    #return nms_bboxes #返回带有class,坐标等信息的框
    return new_img

def imgRestore(image, bboxes, gt_classes_index=None, classes=pc.get_classes()):
    Yinxielist = list() #隐写数据，一会儿转np
    Yinxielist.append(0) #计数初始化为0

    num_classes = len(classes)
    image_h, image_w, channel = image.shape
    fontScale = 0.5
    bbox_thick = int(0.6 * (image_h + image_w) / 600)
    imglist = list()
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))

    fontScale = 0.5
    bbox_thick = int(0.6 * (image_h + image_w) / 600)
    for i, bbox in enumerate(bboxes): #对于每个检测框吧。
         coor = np.array(bbox[:4], dtype=np.int32) #检测框的两个对角吧。。
         if gt_classes_index == None:
            class_index = int(bbox[5]) #class   
            score = bbox[4] #概率吧
         else:
            class_index = gt_classes_index[i]
            score = 1
         bbox_color = colors[class_index]
         classes_name = classes[class_index]
         if classes_name =='sports ball':
            temp = image[coor[1]:coor[3],coor[0]:coor[2]]
            imglist.append(temp)
            r=np.array([0.343, 0.432, 0.63 ,3.769 ,3.82, 3.8, 0.1 ,1])#密钥
            x0 = temp[:, :, 0]
            x1 = temp[:, :, 1]
            x2 = temp[:, :, 2]
            e0 = jiami(x0, r)
            e1 = jiami(x1, r)
            e2 = jiami(x2, r)
            imgcut = np.dstack((e0, e1, e2)) #
            image[coor[1]:coor[3],coor[0]:coor[2]] = imgcut  #换
            Yinxielist.append(coor[0])  
            Yinxielist.append(coor[1])
            Yinxielist.append(coor[2])
            Yinxielist.append(coor[3])

            #Yinxielist[4*i+1] = coor[0]
            #Yinxielist[4*i+2] = coor[1]
            #Yinxielist[4*i+3] = coor[2]
            #Yinxielist[4*i+4] = coor[3]
            fig_name = f'result_jiami.jpg'
            cv2.imwrite(fig_name, image)
    Yinxielist[0] = int((len(Yinxielist) -1)/4)
    print("yinxiedata",Yinxielist)
    newimg = yinxie(image,Yinxielist)
    return newimg

         
    

