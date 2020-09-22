#!/usr/bin/env python
# coding: utf-8
# 导入库并定义辅助功能。 define auxiliary functions.
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import imutils

######################################################
# 下载和准备好YOLO的必要文件，初始化参数
######################################################

# 置信度阈值
confThreshold = 0.25
# 非最大抑制阈值, Non-maximum suppression threshold, NMS
nmsThreshold = 0.45

# 网络输入图像的宽度, Width of network's input image
inpWidth = 416

# 网络输入图像的高度, Height of network's input image
inpHeight = 416

# 加载模型和类。
# 加载类别名称，目前只有一个类别，table
classesFile = "vocTablas.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# 给出模型的配置文件和权重文件，并使用它们加载网络。
modelConfiguration = "tablasFinaltest416320.cfg";
modelWeights = "tablasFinaltrain_10000.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


def getOutputsNames(net):
    """
    获取输出层的名称
    :param net:
    :return:
    """
    # 获取网络中所有层的名称
    layersNames = net.getLayerNames()
    # for i in net.getUnconnectedOutLayers():
    #    print(i[0]-1)
    # 获取输出图层的名称，即the layers with unconnected outputs, ['yolo_82', 'yolo_94', 'yolo_106']
    layers = [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return layers


def drawPred(frame, classId, conf, left, top, right, bottom):
    """
    绘制预测的边界框,  bounding box
    :param frame:
    :param classId:
    :param conf:
    :param left:
    :param top:
    :param right:
    :param bottom:
    :return:
    """
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    #置信度
    label = '%.2f' % conf

    # 获取类别名称及其可信度的标签
    if classes:
        assert (classId < len(classes))
        #label变成 'table:0.99'
        label = '%s:%s' % (classes[classId], label)

    # 在边框上方显示标签
    labelSize, baseLine = cv.getTextSize(text=label, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
    top = max(top, labelSize[1])
    # cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


def postprocess(frame, outs):
    """
    使用非极大值抑制和最低置信度， 删除边界框
    :param frame:
    :param outs:
    :return:
    """
    # 获取图片的实际尺寸， 792 x 612
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # 扫描从网络输出的所有边界框，并仅保留具有高置信度得分的边界框。
    # 将box's类别标签分配为得分最高的类别。
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            #如果大于最低置信度的阈值，才保留
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # 执行NMS非极大抑制，以消除具有较低置信度的冗余重叠框,最终画出一个边界框
    indices = cv.dnn.NMSBoxes(bboxes=boxes, scores=confidences, score_threshold=confThreshold, nms_threshold=nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
    return boxes



# 从框架创建4D blob。
# frame = cv.resize(frame,(inpWidth, inpHeight))
def predict(name):
    """
    预测图像
    # 读取图像的输入。
    # 神经网络的输入图像必须采用称为blob的某种格式。
    # 从输入图像或视频流中读取帧后，将其通过blobFromImage函数传递，
    # 以将其转换为神经网络的输入blob。
    # 在此过程中，它使用1/255的缩放因子将图像像素值缩放到0到1的目标范围。
    # 它还可以将图像调整为给定大小（416，416），而不会裁切。
    # 请注意，这里我们不执行任何均值减法，因此将[0,0,0]传递给函数的均值参数，并将swapRB参数保持为其默认值1。
    # 将输出Blob作为其输入传递到网络中，并运行前向传递以获取预测的边界框列表作为网络的输出。
    # 这些框经过后处理步骤，以过滤掉置信度得分低的框。
    # 我们在左上角打印出每个帧的推理时间。
    # 然后将带有最终边界框的图像保存为磁盘，或者作为用于图像输入的图像，或者使用视频写入器作为输入视频流。
    :param name:
    :return: 返回bounding box 的坐标位置
    """
    #读取图像
    frame = cv.imread(name)
    # blob 归一化， [batch_size, RGB_channel:3, W, H]
    blob = cv.dnn.blobFromImage(image=frame, scalefactor=1 / 255, size=(inpWidth, inpHeight), mean=[0, 0, 0], swapRB=1, crop=False)

    # 设置网络输入
    net.setInput(blob)

    # 运行前向传递以获取输出层的输出
    outs = net.forward(getOutputsNames(net))

    # 移除低置信度的边界框,
    boxes1 = postprocess(frame, outs)
    # 放入efficiency信息。 函数getPerfProfile返回推理的总时间（t）和每个图层的计时（以layersTimes为单位）
    t, _ = net.getPerfProfile()
    # label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    # cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    #画出预测的bounding box
    cv.imwrite("prediction.jpg", frame)
    return boxes1


def showImage(image):
    if len(image.shape) == 3:
        img2 = image[:, :, ::-1]
        plt.imshow(img2)
        plt.show()
    else:
        img2 = image
        plt.imshow(img2, cmap='gray')
        plt.show()


if __name__ == '__main__':
    predict("/Users/admin/git/fine-tuning/images/Latex_100058.jpg")
