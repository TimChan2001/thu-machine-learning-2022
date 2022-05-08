from sklearn import svm
import numpy as npy
from gensim.models import Word2Vec
import datetime

model = Word2Vec.load("word2vec.model") # load word embedding模型

def text2vec(text): # word的list转向量
    sentence = []
    complement_vec = [] # 0向量，用于填充
    for i in range(50):
        complement_vec.append(0)
    idx = 0
    while len(sentence)<100: # 100标准长度，长截短补
        try:
            sentence.append(model.wv[text[idx]].tolist())
            idx+=1
        except:
            idx+=1
        if idx >= len(text):
            break
    while len(sentence)<100:
        sentence.append(complement_vec)
    sentence = npy.array(sentence)
    return sentence.reshape([1,5000])[0] # 转成5000维向量
    

def main(): 
    stopwords = []
    stopwordsfile = open('./stopwords.txt','r') # 从stopwords.txt中parse出停用词
    for line in stopwordsfile:
        line = line.strip()
        stopwords.append(line)

    train_item = []
    test_item = []
    division = npy.load("./division.npy",allow_pickle=True) # 获取训练集和数据集的切分信息
    for i in range(len(division)):
        if i < 20000:
            test_item.append(division[i])
        else:
            train_item.append(division[i])
    test_item.sort()
    train_item.sort()

    test_vec = []
    test_label = []
    train_vec = []
    train_label = []

    items = open('./exp3-reviews.csv','r')
    idx = 0
    for line in items: # parse文本，逻辑同parse_csv_word_embedding.py
        line = line.strip()
        if (line.split('\t'))[0] != 'overall':
            text = []
            idx+=1
            # print(str(idx))
            train = True
            if idx in test_item:
                train = False
            rating = (line.split('\t'))[0]
            if train: # 加入训练集
                train_label.append(int(rating[0]))
            else: # 加入测试集
                test_label.append(int(rating[0]))
            summary = (line.split('\t'))[-2]
            reviewText = (line.split('\t'))[-1]
            end_punctuation = ['.','?','!','~']
            mid_punctuation = [',','<','>','"','(',')',':',';','&','^','[',']','{','}','|','/','*','#','=','_','+','%',"'"]
            sentence = summary.split()
            for i in range(len(sentence)-1,-1,-1):
                while len(sentence[i])>0 and sentence[i][-1] in mid_punctuation:
                    sentence[i] = sentence[i].strip(sentence[i][-1])
                while len(sentence[i])>0 and sentence[i][0] in mid_punctuation: 
                    sentence[i] = sentence[i].strip(sentence[i][0]) 
                while len(sentence[i])>0 and sentence[i][-1] in end_punctuation:
                    sentence[i] = sentence[i].strip(sentence[i][-1])
                while len(sentence[i])>0 and sentence[i][0] in end_punctuation: 
                    sentence[i] = sentence[i].strip(sentence[i][0])     
                if sentence[i].isdigit() or sentence[i].lower() in stopwords or sentence[i]=='':
                    del sentence[i]
            if len(sentence) > 1 :
                for word in sentence:
                    text.append(word)
            start = 0
            for end in range(len(reviewText)):
                if reviewText[end] in end_punctuation:
                    sentence = reviewText[start:end].split()
                    start = end+1
                    for i in range(len(sentence)-1,-1,-1):
                        while len(sentence[i])>0 and sentence[i][-1] in mid_punctuation:
                            sentence[i] = sentence[i].strip(sentence[i][-1])
                        while len(sentence[i])>0 and sentence[i][0] in mid_punctuation: 
                            sentence[i] = sentence[i].strip(sentence[i][0])   
                        if sentence[i].isdigit() or sentence[i].lower() in stopwords or sentence[i]=='':
                            del sentence[i]
                    if len(sentence) > 1:
                        for word in sentence:
                            text.append(word)
            vec = text2vec(text)
            if train: # 加入训练集
                train_vec.append(vec)
            else: # 加入测试集
                test_vec.append(vec)
    print("size of train set: "+str(len(train_vec)))
    clf = svm.SVC(C=0.7, decision_function_shape='ovr', kernel='rbf',max_iter=100) # 训练模型
    print("start to train!")
    start_time = datetime.datetime.now().timestamp()
    clf.fit(train_vec, train_label)
    end_time = datetime.datetime.now().timestamp()
    print("done training after "+str(round(end_time-start_time))+"s!")
    prediction = clf.predict(test_vec) # 预测
    correct = 0
    difference = 0
    for i in range(len(prediction)):
        if prediction[i] == test_label[i]:
            correct+=1
        if prediction[i] > test_label[i]:
            difference += prediction[i] - test_label[i]
        else:
            difference += test_label[i] - prediction[i]

    print("acc: "+str(round(100*correct/len(prediction)))+"%") # 准确率
    print("average difference: "+str(difference/len(prediction))) # 预测打分和ground truth打分的平均差异

if __name__ == "__main__":
   main()
