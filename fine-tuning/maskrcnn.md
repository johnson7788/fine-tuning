# Fine tuning with the Mask RCNN algorithm.

## Installation 

安装[Mask RCNN library](https://github.com/matterport/Mask_RCNN/)及其所有依赖项。 
建议创建一个虚拟环境并使用以下命令安装依赖项。


```python
pip install -r requirementsMaskRCNN.txt
```

The [requirementsMaskRCNN](../code/mask-rcnn/requirementsMaskRCNN.txt) can be found in the code folder. 

## Folder structure

Create a folder ``maskrcnnmodels``, and 使用以下结构整理数据集。

```bash
maskrcnnmodels
└── mydataset
    ├── annots
    │   ├── 00001.xml
    │   ├── 00002.xml
    │   ├── 00003.xml
    │   ├── ...
    └── images
        ├── 00001.jpg
        ├── 00002.jpg
        ├── 00003.jpg
        ├── ...
```
必须使用Pascal VOC数据集（“annots”文件夹的xml文件）对图像进行注释。 
使用此格式注释图像的工具是[LabelImg](https://github.com/tzutalin/labelImg).。

## Necessary files

为了使用此算法微调模型，必须在文件夹中下载以下文件 ``maskrcnnmodels``:
- [TableBank weights](https://www.dropbox.com/s/dcl53rl3xqndfdx/mask_rcnn_tablebank_cfg_0002.h5?dl=1).
- [Template training file](../code/mask-rcnn/template.py)

## Training
首先，必须修改上一步中下载的模板文件。 
只需在模板文件中搜索文本``<- MODIFY ->``并按照说明进行操作即可。 
要训练模型，请在``maskrcnnmodels``文件夹中执行以下命令：

```python
python template.py
```
模型的权重将存储在名为``model_cfg``的文件夹中。 
您可以使用[Mask RCNN notebook](https://colab.research.google.com/drive/1smseOGcUZZjvMfDHnoW8-ancldz-zpOg)
或[predict file](./code/mask-rcnn/predict.py)只能通过更改权重文件。


