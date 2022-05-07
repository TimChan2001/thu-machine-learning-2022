import numpy as npy

def main(): 
    stopwords = []
    stopwordsfile = open('./stopwords.txt','r') # 从stopwords.txt中parse出停用词
    for line in stopwordsfile:
        line = line.strip()
        stopwords.append(line)

    dic = {}
    ratings = {
        1:0,
        2:0,
        3:0,
        4:0,
        5:0
    }
    items = open('./exp3-reviews.csv','r')
    idx = 0
    for line in items:
        line = line.strip()
        if (line.split('\t'))[0] != 'overall':
            idx+=1
            print(str(idx))
            rating = (line.split('\t'))[0]
            rating = int(rating[0])
            ratings[rating]+=1
            summary = (line.split('\t'))[-2]
            reviewText = (line.split('\t'))[-1]
            end_punctuation = ['.','?','!','~',"'"]
            mid_punctuation = [',','<','>','"','(',')',':',';','&','^','[',']','{','}','|','/','*','#','=','_','+','%','-']
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
            for word in sentence:
                if word in dic.keys():
                    dic[word]+=5.0*(rating==1)+5.0*(rating==2)+2.5*(rating==3)+1.0*(rating==4)+1.0*(rating==5)
                else:
                    dic[word]=5.0*(rating==1)+5.0*(rating==2)+2.5*(rating==3)+1.0*(rating==4)+1.0*(rating==5)
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
                    for word in sentence:
                        if word in dic.keys():
                            dic[word]+=5.0*(rating==1)+5.0*(rating==2)+2.5*(rating==3)+1.0*(rating==4)+1.0*(rating==5)
                        else:
                            dic[word]=5.0*(rating==1)+5.0*(rating==2)+2.5*(rating==3)+1.0*(rating==4)+1.0*(rating==5)
    dic = sorted(dic.items(),key=lambda x:x[1],reverse=True)
    dic = dic[0:1000]
    print(dic)
    print(ratings)
    dic_final = {}
    for i in range(len(dic)):
        dic_final[dic[i][0]] = i
    npy.save('word_dict.npy',dic_final)

if __name__ == "__main__":
   main()
