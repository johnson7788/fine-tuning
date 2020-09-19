# Fine tuning with the SSD algorithm.

## Installation 

安装[GluonCV toolkit](https://gluon-cv.mxnet.io/)及其所有依赖项。 
建议创建一个虚拟环境并使用以下命令安装依赖项。

```python
pip install -r requirementsSSD.txt
```

The [requirementsSSD](../code/ssd/requirementsSSD.txt) can be found in the code folder. 

## Folder structure
创建一个文件夹``ssdmodels``，并在其中创建一个名为datasets的文件夹，
并使用以下结构组织数据集。 重要的是数据集的名称必须以前缀``VOC``开头。

```bash
ssdmodels
├── configs
│   └── mydataset_config.py
└── datasets
    └── VOCmydataset
        ├── Annotations
        │   ├── 00001.xml
        │   ├── 00002.xml
        │   ├── 00003.xml
        │   ├── ...
        ├── ImageSets
        │   └── Main
        │       └── train.txt
        └── JPEGImages
            ├── 00001.jpg
            ├── 00002.jpg
            ├── 00003.jpg
            ├── ...
```
必须使用Pascal VOC数据集（``Annotations`` 文件夹的xml文件）对图像进行注释。 
使用此格式注释图像的工具是[LabelImg](https://github.com/tzutalin/labelImg)。 
train.txt文件必须是在JPEGImages文件夹中可用的图像列表（无扩展名）。 
最后，文件``mydataset_config.py``包含配置。 特别是，必须提供以下选项：

```python
classes = ['table']  
datasetName = 'mydataset'
nepochs = 200
```
您只需要使用数据集的名称更改``mydataset``即可。

## Necessary files
为了使用此算法对模型进行微调，必须在``ssdmodels``文件夹中下载以下文件：
- [TableBank weights](https://www.dropbox.com/s/x95ipfjqoncrzt4/ssd_512_resnet50_tablebank_19.params?dl=0).
- [Dataset builder](../code/ssd/finetune_detection_transfer.py)

## Training

您可以使用以下命令训练模型。

```bash
python finetune_detection_transfer.py -c mydataset_config
```
模型的权重将存储在执行上述指令的相同文件夹中。 
您可以使用[SSD notebook](https://colab.research.google.com/drive/1s8xoKf1gk0Aqs324genSCXXNVG3R-wJc) 
或the [predict file](./code/ssd/predict.py)在其中创建的模型。



