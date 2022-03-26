实验环境 `python>=3.7`

1.添加课程提供的`data`和`label`文件夹到根目录下（即所有`.py`所在的目录）

2.运行`parse_email.py`

```
python parse_email.py arg1 arg2
```

程序会提取`./data`中名称在range(arg1,arg2)的文件夹作为测试集，其余作为训练集提取特征，在`./dicts`目录下生成`exist_dict{arg1}_{arg2-1}.npy`和`exist_spam_dict{arg1}_{arg2-1}.npy`，如

```
python parse_email.py 51 76
```

将在`./dicts`目录下生成`exist_dict51_75.npy`和`exist_spam_dict51_75.npy`。程序同时输出已经parse作为训练集的文件夹名称。

3.运行`select_word.py`

```
python select_word.py arg1 arg2
```

程序会读取上一步在`./dicts`目录下生成的`.npy`文件，精炼特征后在`./dicts`目录下生成`words.npy`，同时会输出训练集中垃圾邮件的数量、邮件的总数和提取特征的数量。

4.运行`test.py`

```
python test.py arg1 arg2
```

程序会读取上一步在`./dicts`目录下生成的`words.npy`文件并开始测试，输出正在计算的测试集文件夹名称和当前累计的各项指标，本步骤耗时较长。

若要进行`sender email addr`特征的模型的测试，参照上述相同的步骤运行`parse_email_sender.py`、`select_word_sender.py`和`test_sender.py`即可。