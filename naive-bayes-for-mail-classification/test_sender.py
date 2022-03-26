import os, sys, validators
import numpy as np
"""
邮件中sender email addr特征的测试
"""
def main(argv):# 代码注释见同目录下的test.py，逻辑相同
    spam_total = 0
    ham_total = 0
    predict_spam_correct = 0
    predict_ham_correct = 0
    predict_spam_wrong = 0
    predict_ham_wrong = 0
    train = np.load('./dicts/words_sender.npy',allow_pickle=True)
    exist_dic = np.load('./dicts/exist_sender_dict'+sys.argv[1]+'_'+str(int(sys.argv[2])-1)+'.npy',allow_pickle=True)
    exist_spam_dic = np.load('./dicts/exist_sender_spam_dict'+sys.argv[1]+'_'+str(int(sys.argv[2])-1)+'.npy',allow_pickle=True)
    if len(sys.argv)!=3:
        print('usage: test_sender.py start_test_dir end_test_dir')
        sys.exit()
    label = open('./label/index','r') 
    labels = {}
    for line in label:
        line = line.strip()
        line = line.split()
        if line[0]=='spam':
            labels[line[1]] = 1
        else: 
            labels[line[1]] = 0
    label.close()
    for i in range(int(sys.argv[1]),int(sys.argv[2])):
        dirname=''
        if i < 10:
            dirname='00'+str(i)
        elif i < 100:
            dirname='0'+str(i)
        else:
            dirname=str(i)
        print('./data/'+dirname)
        for filename in range(len(os.listdir(r"./data/"+dirname))): 
                if filename < 10:
                    filename='00'+str(filename)
                elif filename < 100:
                    filename='0'+str(filename)
                else:
                    filename=str(filename)
                file = open('./data/'+dirname+'/'+filename,'r') 
                path = '../data/'+dirname+'/'+filename
                dict_of_words = {}
                dict_of_words['label_of_the_email']=labels[path]
                content=False
                try:
                    for line in file:
                        line = line.strip()
                        if line == '':
                            content = True
                        words = line.split()
                        if len(words)>0 and words[0] == 'From:' and content == False:
                            words.pop(0)                    
                            punctuation = [',','.','?','!','<','>','"',"'",'(',')','-',':',';','&','~','^','[',']','{','}','|','/','*','#','=','_','+','%','@']
                            for word in words:
                                while word != '' and word[0] in punctuation:
                                    word = word.strip(word[0])
                                while word != '' and word[-1] in punctuation:
                                    word = word.strip(word[-1])
                                jump = False
                                if word != '':
                                    if validators.email(word):
                                        words = word.split('.')
                                        for idx in range(len(words)):
                                            for letter in words[idx]:
                                                if letter == '@':
                                                    words.pop(idx)
                                                    jump = True
                                                    break
                                            if jump:
                                                break
                                        for ele in words:
                                            if ele in dict_of_words.keys():
                                                dict_of_words[ele]+=1
                                            else:
                                                dict_of_words[ele]=1    
                    train_data={}
                    exist_dict={}
                    exist_spam_dict={}
                    for key in train.item().keys():
                        train_data[key]=train.item().get(key)
                    for key in exist_dic.item().keys():
                        exist_dict[key]=exist_dic.item().get(key)
                    for key in exist_spam_dic.item().keys():
                        exist_spam_dict[key]=exist_spam_dic.item().get(key) 

                    p_spam = exist_dict['label_of_the_email']/exist_dict['count']
                    p_ham = 1 - p_spam
                    p_x_spam = 1
                    p_x_ham = 1
                    alpha = 1
                    for key in dict_of_words.keys():
                        if key!='label_of_the_email':
                            if key in exist_spam_dict.keys() and key in train_data.keys():
                                p_x_spam*=(exist_spam_dict[key]+alpha)/(exist_dict['label_of_the_email']+alpha*exist_dict['count'])
                                p_x_ham*=(exist_dict[key] - exist_spam_dict[key]+alpha)/(exist_dict['count']-exist_dict['label_of_the_email']+alpha*exist_dict['count'])
                            else:
                                p_x_spam*=alpha/(exist_dict['label_of_the_email']+alpha*exist_dict['count'])
                                p_x_ham*=alpha/(exist_dict['count']-exist_dict['label_of_the_email']+alpha*exist_dict['count'])
                    p_is_spam = p_spam*p_x_spam/(p_spam*p_x_spam+p_x_ham*p_ham)
                    if p_is_spam > 0.7:
                        if dict_of_words['label_of_the_email']==1:
                            predict_spam_correct+=1
                        else:
                            predict_spam_wrong+=1
                    if p_is_spam <= 0.7 :
                        if dict_of_words['label_of_the_email']==0:
                            predict_ham_correct+=1
                        else:
                            predict_ham_wrong+=1
                    if dict_of_words['label_of_the_email']==1:
                        spam_total+=1
                    else:
                        ham_total+=1
                    file.close()
                except:
                    continue
        print('accuracy: '+str((predict_spam_correct+predict_ham_correct)/(ham_total+spam_total)))     
        print('precision: '+str(predict_spam_correct/(predict_spam_correct+predict_spam_wrong)))
        print('recall: '+str(predict_spam_correct/spam_total))
        print('false positive rate: '+str(predict_spam_wrong/ham_total))
        print('false negative rate: '+str(predict_ham_wrong/spam_total))
if __name__ == "__main__":
   main(sys.argv[1:])
