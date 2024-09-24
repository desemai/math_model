# README
第二十一届“华为杯”数学竞赛C题

## 项目介绍
本项目主要使用 *SVM*、*GA*、*AutoGluon* 等方法解决数学建模的多个问题。第三题并未使用本项目的代码进行处理，这里只提供了一部分数据处理的思路，实际的论文中使用了其他方法。  

## 项目依赖
建议使用 *conda* 等虚拟化环境安装依赖。  
```conda create -n math python=3.10```  
```pip install -r requirements.txt```  
需要注意的是，这里还有一些依赖包并没有安装，有的包比较常见，直接使用 *pip* 进行安装即可。  
[AutoGluon](https://auto.gluon.ai)需要一些特殊的依赖，请参考官方的文档进行对应的安装。

## 项目结构
```
.
├── agModels_Regression  # 模型文件夹
├── analyse.py           # 第三题代码（未使用）
├── automl.ipynb         # 第四题部分代码
├── classify.py          # 第一题代码
├── convert.py           # 第四题部分代码
├── data                 # 原始数据
├── data_convert         # 转换后的数据
├── multi_optim.py       # 第五题代码
├── optim.py             # 第二题代码
├── README.md
├── requirements.txt
└── src                  # 部分结果储存地址
```


## 致谢
非常感谢一起奋战的两位同学，没有他们的支持，该项目也不能及时完成。