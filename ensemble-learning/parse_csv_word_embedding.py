from gensim.models import Word2Vec
import numpy as npy

stopwords = []
text = []
stopwordsfile = open('./stopwords.txt','r') # 从stopwords.txt中parse出停用词
for line in stopwordsfile:
    line = line.strip()
    stopwords.append(line)

ratings = []
summaries = []
reviewTexts = []

items = open('./exp3-reviews.csv','r')
idx = 0
for line in items:
    print(str(idx))
    idx+=1
    line = line.strip()
    if (line.split('\t'))[0] != 'overall':
        rating = (line.split('\t'))[0]
        summary = (line.split('\t'))[-2]
        reviewText = (line.split('\t'))[-1]
        ratings.append(rating)
        summaries.append(summary)
        reviewTexts.append(reviewText)
        end_punctuation = ['.','?','!','~']
        mid_punctuation = [',','<','>','"','(',')',':',';','&','^','[',']','{','}','|','/','*','#','=','_','+','%',"'"]
        start = 0
        text_len = 0
        for end in range(len(reviewText)):
            if reviewText[end] in end_punctuation:
                sentence = reviewText[start:end].split() # 先通过end_punctuation拆分出句子，再以句子为单位分词
                start = end+1
                for i in range(len(sentence)-1,-1,-1):
                    while len(sentence[i])>0 and sentence[i][-1] in mid_punctuation:
                        sentence[i] = sentence[i].strip(sentence[i][-1])
                    while len(sentence[i])>0 and sentence[i][0] in mid_punctuation: 
                        sentence[i] = sentence[i].strip(sentence[i][0])   
                    if sentence[i].isdigit() or sentence[i].lower() in stopwords or sentence[i]=='':
                        del sentence[i]
                if len(sentence) > 1: # 去掉只有一个词的句子，因为没有上下文信息
                    text.append(sentence) # 所有句子存入text中
                    text_len+=len(sentence) # 统计文本长度，便于确定标准向量长度
        sentence = summary.split() # 整个summary作为一个句子
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
        if len(sentence) > 1 :# 去掉只有一个词的句子，因为没有上下文信息
            text.append(sentence) # 所有句子存入text中
            text_len+=len(sentence) # 统计文本长度，便于确定标准向量长度

print("sentences: "+str(len(text)))
sentences = text
model = Word2Vec(min_count=10,vector_size=50)
model.build_vocab(sentences)
model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs) # train
# model.save("word2vec.model") # 保存模型




        