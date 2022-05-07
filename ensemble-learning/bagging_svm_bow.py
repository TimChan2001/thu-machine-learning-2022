from sklearn import svm
import numpy as npy
import datetime,random

dict = npy.load("word_dict.npy",allow_pickle=True)

def text2vec(text):
    vec= []
    for i in range(1000):
        vec.append(0)
    for i in range(len(text)):
        if dict.item().get(text[i]) != None:
            vec[dict.item().get(text[i])]+=1
    return vec
    
def main(): 
    stopwords = []
    stopwordsfile = open('./stopwords.txt','r') # 从stopwords.txt中parse出停用词
    for line in stopwordsfile:
        line = line.strip()
        stopwords.append(line)

    train_item_all = []
    test_item = []
    test_vec = []
    test_label = []
    train_vec_all = []
    train_label_all = []
    predictions = []
    division = npy.load("./division.npy",allow_pickle=True)
    for i in range(len(division)):
        if i < 20000:
            test_item.append(division[i])
        else:
            train_item_all.append(division[i])
    test_item.sort()
    train_item_all.sort()
    items = open('./exp3-reviews.csv','r')
    idx = 0
    for line in items:
        line = line.strip()
        if (line.split('\t'))[0] != 'overall':
            text = []
            idx+=1
            # print(str(idx))
            train = False
            test = False
            if idx in test_item:
                test = True
            if idx in train_item_all:
                train = True
            rating = (line.split('\t'))[0]
            if train:
                train_label_all.append(int(rating[0]))
            if test:
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
                train_vec_all.append(vec)
            if test:
                test_vec.append(vec)
    for model_idx in range(30):
        random.seed(datetime.datetime.now().timestamp())
        random.shuffle(train_item_all)
        train_item = train_item_all[0:20000]
        train_vec = []
        train_label = []
        for k in range(len(train_item_all)):
            if k+1 in train_item:
                train_vec.append(train_vec_all[k+1])
                train_label.append(train_label_all[k+1])
        print("size of train set: "+str(len(train_vec)))
        clf = svm.SVC(C=0.7, decision_function_shape='ovr', kernel='rbf',max_iter=500)
        print("start to train model "+str(model_idx)+"!")
        start_time = datetime.datetime.now().timestamp()
        clf.fit(train_vec, train_label)
        end_time = datetime.datetime.now().timestamp()
        print("done training model "+str(model_idx)+" after "+str(round(end_time-start_time))+"s!")
        prediction = clf.predict(test_vec)
        predictions.append(prediction)
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
            vote[predictions[j][i]]+=1
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