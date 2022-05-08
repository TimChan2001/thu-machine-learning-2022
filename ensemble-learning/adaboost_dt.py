from sklearn import tree
import numpy as npy
from gensim.models import Word2Vec
import datetime
# 注释见adaboost_svm.py
model = Word2Vec.load("word2vec.model")

def text2vec(text):
    sentence = []
    complement_vec = []
    for i in range(50):
        complement_vec.append(0)
    idx = 0
    while len(sentence)<100:
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
    return sentence.reshape([1,5000])[0]
    

def main(): 
    stopwords = []
    stopwordsfile = open('./stopwords.txt','r') # 从stopwords.txt中parse出停用词
    for line in stopwordsfile:
        line = line.strip()
        stopwords.append(line)

    train_item = []
    test_item = []
    division = npy.load("./division.npy",allow_pickle=True)
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
    for line in items:
        line = line.strip()
        if (line.split('\t'))[0] != 'overall':
            text = []
            idx+=1
            # print(str(idx))
            train = True
            if idx in test_item:
                train = False
            rating = (line.split('\t'))[0]
            if train:
                train_label.append(int(rating[0]))
            else:
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
            if train:
                train_vec.append(vec)
            else:
                test_vec.append(vec)
    print("size of train set: "+str(len(train_vec)))
    sample_weight = []
    model_weight = []
    predictions = []
    learning_rate = 0.5
    for i in range(len(train_vec)):
        sample_weight.append(1/len(train_vec))
    for iter in range(15):
        clf = tree.DecisionTreeClassifier()
        print("start to train!")
        start_time = datetime.datetime.now().timestamp()
        clf.fit(train_vec, train_label, sample_weight=sample_weight)
        end_time = datetime.datetime.now().timestamp()
        print("done training model "+str(iter)+" after "+str(round(end_time-start_time))+"s!")
        predictions.append(clf.predict(test_vec))
        prediction = clf.predict(train_vec)
        error_rate = 0
        for i in range(len(prediction)):
            if prediction[i] != train_label[i]:
                error_rate+=sample_weight[i]
        print("e: "+str(error_rate))
        alpha = npy.round(learning_rate*(npy.log((1-error_rate)/error_rate) + npy.log(5 - 1)),8)
        model_weight.append(alpha)
        for i in range(len(prediction)):
            sample_weight[i]*=npy.exp(alpha*(prediction[i] != train_label[i]))
        sample_weight/=sum(sample_weight)
    prediction_final = []
    for i in range(len(predictions[0])):
        vote = {
            1:0,
            2:0,
            3:0,
            4:0,
            5:0
        }
        for j in range(len(predictions)):
            vote[predictions[j][i]]+=model_weight[j]
        votes = 0
        winner = 0
        for j in range(1,6):
            if vote[j] >= votes:
                votes = vote[j] 
                winner = j
        prediction_final.append(winner)
        
    correct = 0
    difference = 0
    for i in range(len(prediction_final)):
        if prediction_final[i] == test_label[i]:
            correct+=1
        if prediction_final[i] > test_label[i]:
            difference += prediction_final[i] - test_label[i]
        else:
            difference += test_label[i] - prediction_final[i]

    print("acc: "+str(round(100*correct/len(prediction_final)))+"%")
    print("average difference: "+str(difference/len(prediction_final)))

if __name__ == "__main__":
   main()
