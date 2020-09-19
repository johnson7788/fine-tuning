# Fine tuning with the YOLO algorithm.

## Installation 

安装 [Darknet library](https://pjreddie.com/darknet/yolo/):

```python
git clone https://github.com/AlexeyAB/darknet
cd darknet
make
``` 

## Folder structure

在``darknet``文件夹内创建一个文件夹 ``datasets``，
并在其中创建一个名为``mydataset``的文件夹，并使用以下结构组织数据集。 重要的是，数据集的名称必须以前缀``VOC``开头。

```bash
darknet
└── datasets
    └── mydataset
        └── train
            └── JPEGImages
                ├── 00001.jpg
                ├── 00001.xml
                ├── 00002.jpg                
                ├── 00002.xml
                ├── 00003.jpg                                
                ├── 00003.xml
                └── ...
```

图像必须使用Pascal VOC数据集进行注释。 使用此格式注释图像的工具是[LabelImg](https://github.com/tzutalin/labelImg). 
最初，所有图像和注释必须存储在同一文件夹中。 我们需要使用以下python脚本将图像转换为YOLO格式
需要[pascal2yolo file](../code/yolo/pascal2yolo_1class.py)，该文件应下载到``yolomodels``文件夹中：

```bash
python pascal2yolo_1class.py -d datasets/mydataset/train/JPEGImages 
```

This produces the following file structure
```bash
darknet
└── datasets
    └── mydataset
        └── train
            └── JPEGImages
                ├── 00001.jpg
                ├── 00001.txt
                ├── 00001.xml
                ├── 00002.jpg
                ├── 00002.txt
                ├── 00002.xml
                ├── 00003.jpg                                
                ├── 00003.txt
                ├── 00003.xml
                └── ...
```
您必须将生成的.txt文件移动到``labels`` 文件夹中：

```bash
darknet
└── datasets
    └── mydataset
        └── train
            ├── JPEGImages
            │   ├── 00001.jpg
            │   ├── 00001.xml
            │   ├── 00002.jpg
            │   ├── 00002.xml
            │   ├── 00003.jpg                                
            │   ├── 00003.xml
            │   └── ...
            │   
            └── labels
                ├── 00001.txt
                ├── 00002.txt
                ├── 00003.txt  
                └── ...
```

最后，您需要生成带有文件列表的txt。 在darknet文件夹中，执行以下命令。

``bash
find `pwd`/datasets/mydataset/train/JPEGImages/*.jpg > datasets/mydataset/train.txt
``

## Necessary files
为了使用该算法对模型进行微调，有必要在 ``yolomodels``文件夹中下载以下文件：
- [TableBank weights](https://www.dropbox.com/s/jbgosn1t83h1bqi/tablasFinaltrain_10000.weights?dl=1).
- [Configuration file](../code/yolo/tablasFinaltrain416320.cfg)
- [Names file](../code/yolo/vocTablas.names)
- [Data file](../code/yolo/tablasFinal.data)

如果为数据集命名, 文件夹``mydataset``，则需要修改[data file](../code/yolo/tablasFinal.data), 为您的数据集的名称。


## Training

您可以通过使用darknet文件夹中的以下命令来训练模型。

```bash
darknet detector train tablasFinal.data tablasFinaltest416320.cfg tablasFinaltrain_10000.weights
```
模型的权重将存储在 ``darknet``目录的 ``backup`` 文件夹中。
您可以使用[YOLO notebook](https://colab.research.google.com/drive/19x3FL2vUjF0as6CKrYKmjrqsiiUTjkw6)在其中创建的模型
或[predict file](./code/yolo/predict.py).




