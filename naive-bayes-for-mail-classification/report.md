实验1 - 朴素贝叶斯实验报告

陈意扬 计96 2019011341

实验目的

实现一个基于贝叶斯算法的垃圾电子邮件分类器

在真实数据集上评估分类器性能

分析实验结果



实验原理

本次实验所基于的实验原理为贝叶斯学习中的朴素贝叶斯方法。模型中把每封电子邮件视为一个样本点`<xn,yn>`，`xn`为抽取的特征向量，`yk`为`label`，在{spam，ham}中取值。贝叶斯公式为
$$
P(y_k=\omega_i|x_k)=\frac{P(x_k|y_k=\omega_i)P(\omega_i)}{P(x_k)}=\frac{P(x_k|y_k=\omega_i)P(\omega_i)}{\Sigma_jP(x_k|y_k=\omega_j)P(\omega_j)}
$$

$$
P(\omega_i)表示该样本点属于第i类别的先验概率。本问题中i只有两类，spam和pam。~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$$

朴素贝叶斯模型对特征有独立性假设，则对于n维特征向量`xk`有
$$
P(x_k|y_k=\omega_i)=\Pi_{j=1}^nP(x_k^{(j)}|y_k=\omega_i)
$$

$$
P(y_k=\omega_i|x_k)=\frac{P(\omega_i)\Pi_{j=1}^nP(x_k^{(j)}|y_k=\omega_i)}{\Sigma_m(P(\omega_m)\Pi_{j=1}^nP(x_k^{(j)}|y_k=\omega_m))}
$$

贝叶斯分类器把后验概率最大的一类作为测试样本的分类，当label只有两个时，则取大于（等于）0.5概率的类别为分类。然而实际上，由于假阴性和假阳性错误分类的代价不同，当spam贝叶斯概率大于0.7时我才将测试样本邮件判定为spam。

实验步骤

1.切分数据集并提取特征（parse_email.py）

实验数据集来自课程提供的The English E-mail Data set。总共37822封email被分装在127个文件夹中。实验要求采用交叉验证，所以我将数据集切分为5份（按文件夹名排序000~025、026~050、051~075、076~100和101~126）,轮流取其中一份为测试集，其余四份为训练集。在实际parse过程中部分邮件无法解码，实际使用到的邮件有32401封，其中垃圾邮件有20030封。
通过对邮件格式的观察我发现在邮件头和邮件内容之间有一个空行，通过识别空行我将每封邮件分为header和content两个部分。在header中通过对'Subject:'的识别提取出title。我认为相同的词在title中和在content中属于不同的特征，所以在抽取bag of words特征的时候将content中的word和title中的word通过加上后缀分开。
同时我将所有的数字都归类为“number”特征、所有的链接归类为“url”特征、所有的email addr归类为“email”特征，并根据它们出现在title还是content中加上后缀。对于字母和数字混杂的词，我使用eliminate_num_from_str方法将字母词干提取出来作为一个word并同时记录一个“number”特征，所有words都去除了首尾的标点符号。
参考文本分类的general做法我选择去掉words中的停用词。我使用了之前的人工智能导论课实验作业的停用词列表（stopwords.txt），将已提取出来的words小写化之后和停用词表比对来去除停用词。
经过上述抽取后每个训练集中term的数量在50w左右（使用交叉验证方法循环选取训练集，总共有5个训练集，图中命令行两个参数为测试集的文件夹范围，其余文件作为训练集），存储在exist_dict{test_set_begin_number}_{}.npy和exist_spam_dict{}_{}.npy中，这里我抽取的特征是每个word在邮件中的存在情况，即记录一个词在多少封邮件中出现。我没有选择word出现的频率，因为我认为记录频率存在较大bias（同一个word在一封邮件中反复出现，这样的word频率大然而对邮件分类提供的信息很少）。


2.精炼特征（select_word.py）

50w的term规模太大，我在select_word.py对前一阶段得到的npy文件进行words的精炼。我首先做了频次截断，去掉所有出现在小于3封邮件的words。之后，我在ens2(word, dict, dict2)方法中计算经每个word分类后（有这个word的邮件和没有这个word的邮件）的信息增益，按信息增益的大小筛掉后2%的words。
经过特征精炼后，每个训练集中term的数量减少到6w左右

3.对测试集进行测试（test.py）

首先对测试集进行特征提取（方法和对训练集的特征提取类似）

模型的评价

ISSUE 1: THE SIZE OF TRAINING SET

ISSUE 2: ZERO-PROBABILITIES

ISSUE 3: SPECIFIC FEATURES

