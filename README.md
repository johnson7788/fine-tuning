# 用于表检测的Close-Domain微调

在这个项目中，我们展示了使用  [TableBank dataset](https://github.com/doc-analysis/TableBank),
Close-Domain训练的模型对表检测模型进行微调的好处。此外，我们提供了使用已构建模型的所有工具，并使用自定义数据集，微调了新的检测模型。 


## TableBank

我们从训练TableBank数据集的几个模型开始。 所有模型，结果和工具在 [TableBank page](TableBank.md) 

## Colab Notebooks for prediction
你可以使用以下notebooks用于预测模型

- [MaskRCNN Notebook](https://colab.research.google.com/drive/1smseOGcUZZjvMfDHnoW8-ancldz-zpOg)
- [RetinaNet Notebook](https://colab.research.google.com/drive/1Zgu7v7jLAKe-xITDbhBe9EDdCUozW-OB)
- [SSD Notebook](https://colab.research.google.com/drive/1s8xoKf1gk0Aqs324genSCXXNVG3R-wJc)
- [YOLO Notebook](https://colab.research.google.com/drive/19x3FL2vUjF0as6CKrYKmjrqsiiUTjkw6)

## Model Zoo for table detection

从使用TableBank数据集构建的模型中，我们可以对不同来源的表检测进行微调。 
关于此过程的所有信息都在[Model Zoo for table detection page](ModelZoo.md)中进行了说明，
与通过自然图像训练的模型相比，该模型显示了应用从TableBank数据集生成的微调模型的好处。

## Fine-tuning
我们提供了必要的工具，以使用TableBank数据集构建的模型作为基础来创建自定义表检测模型。
The instructions are provided in the [Fine-tuning page](FineTuning.md). 

## Citation

Use this bibtex to cite this work:

```
@misc{CasadoGarcia19,
  title={The Benefits of Close-Domain Fine-Tuning for Table Detection in Document Images},
  author={A. Casado-García and C. Domínguez and J. Heras and E. Mata and V. Pascual},
  year={2019},
  note={\url{https://github.com/holms-ur/fine-tuning/}},
}
```
## Acknowledgments
This work was partially supported by Ministerio de Economía y Competitividad [MTM2017-88804-P], Ministerio de Ciencia, Innovación y Universidades [RTC-2017-6640-7], Agencia de Desarrollo Económico de La Rioja [2017-I-IDD-00018], and the computing facilities of Extremadura Research Centre for Advanced Technologies (CETA-CIEMAT), funded by the European Regional Development Fund (ERDF). CETA-CIEMAT belongs to CIEMAT and the Government of Spain.


        
      
