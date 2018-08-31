# 模型组评分系统说明

目前最新的0.1.6版本涵盖了肺部、脑部两个模块，并重构了common.custom_metric中的ClassificationMetric,将多分类指标
按类别存成list列表，并兼容了泛化后的objmatch 0.0.2版本；同时对于脑部模块加入online预测的功能，传入迭代器从而避免一次性读取全部数据
占用大量内存的问题，另外加入了一些新的画图功能和统计指标例如dice(2*tp/(2*tp+fp+fn))、phys_vol_diff/mm^3(gt与pred统计的
分割区域物理体积的差值)；0.1.5在0.1.4基础上重构了每个模块的config文件，将所有的配置参数写成了python的类，当传入具有分类信息的.xls文件路径时
再进行初始化，从而将各个项目模块彻底解耦。

运行肺、心脏、脑部、胸腔等各项目的具体流程请见各项目子文件夹的README.md，运行前请确认各项目所需要的分类信息存放在各项目目录下的
.xls文件中，默认文件名为'classname_labelname_mapping.xls'

## python库的版本依赖

objmatch: networkx <= 2.0

model_eval: pandas >= 0.22.0, pandas != 0.23.0, opencv-python >= 3.3.0.10，matplotlib<=2.1.1

## 超参数调整

### F-Score
对于通过f-score选择最优模型的功能模块(lung, brain等)，在该模块目录下的配置文件config.py中可以设置config.FSCORE_BETA。该参数
为f-score中recall和precision的相对权重，越大则recall相对precision的权重越大，默认值为1.,具体定义见https://en.wikipedia.org/wiki/F1_score

### confidence threshold
对于具有多置信度概率筛选功能的模块(lung, brain等)，在该模块目录下的配置文件config.py中可以设置config.TEST.CONF_THRESHOLD。评估程序
会根据其中的一系列阈值统计模型输出结果。

## 评估指标可视化

### RP曲线

对于肺部结节检测，在RP_plot.py中设置cls_key='class', 并定义好读入文件的路径、文件名，可以生成按各类别画出的曲线，支持.xlsx与.json。

如果想画出按其他关键词划分的曲线，也可以修改相应的cls_key来生成。
```
由模型评估生成的.json文件画图：
python RP_plot.py --json

由模型评估生成的.xlsx文件画图：
python RP_plot.py

```
tip: 对于脑部分割任务，默认的统计方式是按病人号划分的。在RP_plot.py中设置cls_key='PatientID'，可以生成像https://git.infervision.com/w/%E5%87%BA%E8%A1%80%E6%80%A7%E5%8D%92%E4%B8%AD/ 
上以病人为单位统计的RP曲线。如果只是想画所有病人总共的统计曲线，则设置cls_key='class'并运行如下命令：

```
由模型评估生成的.json文件画图：
python RP_plot.py --json --total

由模型评估生成的.xlsx文件画图：
python RP_plot.py --total

```

### ROC 曲线

由于检出任务无法统计tn，从而无法计算fp rate，目前只适用于脑部分割任务。在ROC_plot.py中设置cls_key='class'，定义读入文件的路径、文件名，
可以生成按各类别画出的曲线，支持.xlsx与.json。

如果想画出按其他关键词划分的曲线，也可以修改相应的cls_key来生成。
```
由模型评估生成的.json文件画图：
python ROC_plot.py --json

由模型评估生成的.xlsx文件画图：
python ROC_plot.py
```


