import math, sys, numpy as np
"""
邮件中bag of words特征的精炼
"""
def ens2(word, dict, dict2):  # 计算经过word分类后的信息增益，用于筛选word
    s = dict.item().get('count')
    spam = dict.item().get('label_of_the_email')
    sv1 = dict.item().get(word)
    sv2 = s-sv1
    ensv1 = 1
    ensv2 = 1
    if sv2 == 0:
        return 1 # 没有起到分类效果，直接舍去
    if word not in dict2.item().keys():
        ensv1 = 0 # 提前计算防止除法分母出现0
        spam_in_sv1 = 0
    else:
        spam_in_sv1 = dict2.item().get(word)
    spam_in_sv2 =  spam - spam_in_sv1
    if spam_in_sv2 == 0:
        ensv2 = 0 # 提前计算防止除法分母出现0
    if spam_in_sv1 == sv1:
        ensv1 = 0
    if spam_in_sv2 == sv2:
        ensv2 = 0
    if ensv1 == 1: # 计算经word分类后的熵
        ensv1 = (spam_in_sv1/sv1)*math.log(sv1/spam_in_sv1,2) + ((sv1-spam_in_sv1)/sv1)*math.log(sv1/(sv1-spam_in_sv1),2)
    if ensv2 == 1:
        ensv2 = (spam_in_sv2/sv2)*math.log(sv2/spam_in_sv2,2) + ((sv2-spam_in_sv2)/sv2)*math.log(sv2/(sv2-spam_in_sv2),2)
    ens2 = (sv1/s)*ensv1+(sv2/s)*ensv2
    return ens2

def main(argv):
    if len(sys.argv)!=3:
        print('usage: select_word.py start_test_dir end_test_dir')
        sys.exit()
    dic={}
    dict = np.load('./dicts/exist_dict'+sys.argv[1]+'_'+str(int(sys.argv[2])-1)+'.npy',allow_pickle=True) # load训练集
    print('spam: '+str(dict.item().get('label_of_the_email')))
    print('total: '+str(dict.item().get('count')))
    dict2 = np.load('./dicts/exist_spam_dict'+sys.argv[1]+'_'+str(int(sys.argv[2])-1)+'.npy',allow_pickle=True)

    for key in dict.item().keys():
        if dict.item().get(key)>4: # 去掉所有出现邮件数小于5的词
            dic[key]=dict.item().get(key)
    #print(len(dic.keys()))
    entropy = {}
    for word in dic.keys():
        entropy[word]=ens2(word,dict,dict2)
    entropy = sorted(entropy.items(),key=lambda x:x[1],reverse=False) 
    limit = len(entropy)*0.98 # 去掉信息增益最小的2%的词
    print(limit)
    while len(entropy)>limit:
        entropy.pop()
    after = {}
    for word, en in entropy:
        after[word] = dict.item().get(word)
    np.save('./dicts/words.npy',after) # 筛选词后存储作为模型

if __name__ == "__main__":
   main(sys.argv[1:])
