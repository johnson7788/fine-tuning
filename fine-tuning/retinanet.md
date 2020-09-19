# Fine tuning with the RetinaNet algorithm.

## Installation 

安装[RetinaNet library](https://github.com/fizyr/keras-retinanet) 及其所有依赖项。 
建议创建一个虚拟环境并使用以下命令安装依赖项。

```python
pip install -r requirementsRetinaNet.txt
```

The [requirementsRetinaNet](../code/retinanet/requirementsRetinaNet.txt) can be found in the code folder. 

## Folder structure

Create a folder ``retinanetmodels``, and 使用以下结构整理数据集。

```bash
retinanetmodels
└── mydataset
    ├── annotations
    │   ├── 00001.xml
    │   ├── 00002.xml
    │   ├── 00003.xml
    │   ├── ...
    ├── images
    │   ├── 00001.jpg
    │   ├── 00002.jpg
    │   ├── 00003.jpg
    │   ├── ...
    ├── snapshots
    └── train.txt
```

图像必须使用Pascal VOC数据集（``annotations``文件夹的xml文件）进行注释。 
使用此格式注释图像的工具是[LabelImg](https://github.com/tzutalin/labelImg)。 
 ``train.txt``必须是在``images``文件夹中可用的图像列表（无扩展名）。 
 最后，``snapshots``是一个空文件夹，用于存储模型的权重。

## Necessary files

为了使用此算法微调模型，必须在文件夹中下载以下文件 ``retinanetmodels``:
- [TableBank weights](https://www.dropbox.com/s/rx5zlz3ovywddlh/resnet50_csv_15.h5?dl=1).
- [Dataset builder](../code/retinanet/build_dataset.py)

## Training
首先，有必要通过执行以下操作使用[dataset builder file](../code/retinanet/build_dataset.py)准备数据集：

```python
python build_dataset.py -b mydaset
```

您应该以以下结构结束：
```bash
retinanetmodels
└── mydataset
    ├── annotations
    │   ├── 00001.xml
    │   ├── 00002.xml
    │   ├── 00003.xml
    │   ├── ...
    ├── images
    │   ├── 00001.jpg
    │   ├── 00002.jpg
    │   ├── 00003.jpg
    │   ├── ...
    ├── resnet50_csv_15.h5  
    ├── retinanet_train.csv
    ├── retinanet_classes.csv
    ├── snapshots
    └── train.txt
```

然后，可以通过执行以下步骤来训练模型：

```bash
python retinanet-train --batch-size 4 --steps <steps> --epochs <epochs> --weights resnet50_csv_15.h5 --multi-gpu-force --multi-gpu 2 --snapshot-path mydataset/snapshots csv retinanet_train.csv retinanet_classes.csv
```
有必要通过将要训练模型的图像数除以4来更改``<steps>``的值，
和更改训练的epochs数，即``<epochs>``的值。

模型的权重将存储在``snapshots``文件夹中。 
您可以使用[RetinaNet notebook](https://colab.research.google.com/drive/1Zgu7v7jLAKe-xITDbhBe9EDdCUozW-OB)
或[predict file](./code/retinanet/predict.py) 仅通过更改权重文件，
但首先必须按照[RetinaNet page](https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model)的说明转换模型的权重。



