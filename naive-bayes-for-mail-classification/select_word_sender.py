import math, sys, numpy as np
"""
邮件中sender email addr特征的精炼
"""
def ens2(word, dict, dict2): # 代码注释见同目录下的select_word.py，逻辑相同
    s = dict.item().get('count')
    spam = dict.item().get('label_of_the_email')
    sv1 = dict.item().get(word)
    sv2 = s-sv1
    ensv1 = 1
    ensv2 = 1
    if sv2 == 0:
        return 1 # give up
    if word not in dict2.item().keys():
        ensv1 = 0
        spam_in_sv1 = 0
    else:
        spam_in_sv1 = dict2.item().get(word)
    spam_in_sv2 =  spam - spam_in_sv1
    if spam_in_sv2 == 0:
        ensv2 = 0 # in 
    if spam_in_sv1 == sv1:
        ensv1 = 0
    if spam_in_sv2 == sv2:
        ensv2 = 0
    if ensv1 == 1:
        ensv1 = (spam_in_sv1/sv1)*math.log(sv1/spam_in_sv1,2) + ((sv1-spam_in_sv1)/sv1)*math.log(sv1/(sv1-spam_in_sv1),2)
    if ensv2 == 1:
        ensv2 = (spam_in_sv2/sv2)*math.log(sv2/spam_in_sv2,2) + ((sv2-spam_in_sv2)/sv2)*math.log(sv2/(sv2-spam_in_sv2),2)
    ens2 = (sv1/s)*ensv1+(sv2/s)*ensv2
    return ens2

def main(argv):
    if len(sys.argv)!=3:
        print('usage: select_word_sender.py start_test_dir end_test_dir')
        sys.exit()
    dic={}
    dict = np.load('./dicts/exist_sender_dict'+sys.argv[1]+'_'+str(int(sys.argv[2])-1)+'.npy',allow_pickle=True)
    print('spam: '+str(dict.item().get('label_of_the_email')))
    print('total: '+str(dict.item().get('count')))
    dict2 = np.load('./dicts/exist_sender_spam_dict'+sys.argv[1]+'_'+str(int(sys.argv[2])-1)+'.npy',allow_pickle=True)
    for key in dict.item().keys():
        if dict.item().get(key)>2:
            dic[key]=dict.item().get(key)
    print(len(dic.keys()))
    entropy = {}
    for word in dic.keys():
        entropy[word]=ens2(word,dict,dict2)
    entropy = sorted(entropy.items(),key=lambda x:x[1],reverse=False) 
    limit = len(entropy)*0.9
    print(limit)
    while len(entropy)>limit:
        entropy.pop()
    after = {}
    for word, en in entropy:
        after[word] = dict.item().get(word)
    np.save('./dicts/words_sender.npy',after)

if __name__ == "__main__":
   main(sys.argv[1:])
