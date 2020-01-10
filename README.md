# NER-PyTorch

因最近在做安全领域的NER，所以正好记录一下自己用的BILSTM-CRF、ELMo、BERT等模型来做NER模型的代码

## Bi-LSTM-CharCNN-CRF

##### 数据集CoNLL2003

sh.conll_run.sh 直接运行代码

- conll_run.sh 运行文件
- dataset文件夹

  - conll2003数据集，格式转换成了，text-bieos-bio字典形式
- model.py
  - Bi-LSTM-CharCNN-CRF模型
- util.py  一些数据处理函数、构建embedding向量函数
- span_util.py  
  - BIEOS->BIO 
  - BIO->BIEOS
  - 计算F1的函数（实体级别）restrict-f1首先将预测序列中的实体召回，方便计算准确率召回率F1
- metrics.py

  - 从span_util.py 召回实体后计算准确率召回率F1
- data.py

  - 构建词、字的词典等函数
- crf.py

## BERT-Highway-CRF（BERT相关模型）

有时间了整理一下在放上去吧。待更新

## Other Model......

待更新