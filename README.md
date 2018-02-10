
# 通过机器学习相关算法，检测两个句子的语义相似度

## 1.项目内容：
本次项目提供一系列的英文句子对，每个句子对的两个句子，在语义上具有一定的相似性；每个句子对，获得一个在0-5之间的分值来衡量两个句子的语义相似性，打分越高说明两者的语义越相近。

项目提供数据为txt文件，字段之间以tab分割。

训练数据文件，共有1500个数据样本，共有4个字段；第一个字段为样本编号，第二个字段为一个句子，第三个字段为另一个句子，第四个字段为两个句子的语义相似度打分，如下：

10001	two big brown dogs running through the snow.	A brown dog running through the grass.	2.00000 

10002	A woman is peeling a potato.	A woman is slicing a tomato.	1.33300

测试数据文件，共有750个数据样本，共有3个字段；第一个字段为样本编号，第二个字段为一个句子，第三个字段为另一个句子，
举训练样例说明如下：

10001	two big brown dogs running through the snow.	 A brown dog running through the grass.

10002	A woman is peeling a potato.	 A woman is slicing a tomato.

## 2.基本思路：
参见我们第一周的pdf


## 3.最终成绩
第五名
