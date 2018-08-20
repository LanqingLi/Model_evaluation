# 模型组评分系统说明

运行肺、心脏、脑部、胸腔等各项目的具体流程请见各项目子文件夹的README.md

## python库的版本依赖

objmatch: networkx <= 2.0

model_eval: pandas >= 0.22.0, pandas != 0.23.0

## 评估指标可视化

### RP曲线

在RP_plot中定义读入文件的路径、文件名，支持.xlsx与.json
```
由模型评估生成的.json文件画图：
python RP_plot.py --json

由模型评估生成的.xlsx文件画图：
python RP_plot.py

```
tip: 对于脑部分割任务，在RP_plot.py中设置cls_key='PatientID'可以生成像https://git.infervision.com/w/%E5%87%BA%E8%A1%80%E6%80%A7%E5%8D%92%E4%B8%AD/ 上以病人为单位统计的RP曲线

### ROC 曲线
在ROC_plot中定义读入文件的路径、文件名，支持.xlsx与.json
```
由模型评估生成的.json文件画图：
python ROC_plot.py --json

由模型评估生成的.xlsx文件画图：
python ROC_plot.py
```


