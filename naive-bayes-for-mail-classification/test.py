import os, sys, validators
import numpy as np

"""
eliminate_num_from_str方法从字符串中去除数字
"""
def eliminate_num_from_str(str1):
    str2=''
    num = 0
    for i in str1:
        if i.isdigit():
            num = 1
        else:
            str2 += i
    return str2, num

def main(argv):
    spam_total = 0
    ham_total = 0
    predict_spam_correct = 0
    predict_ham_correct = 0
    predict_spam_wrong = 0
    predict_ham_wrong = 0
    train = np.load('./dicts/words.npy',allow_pickle=True)
    exist_dic = np.load('./dicts/exist_dict'+sys.argv[1]+'_'+str(int(sys.argv[2])-1)+'.npy',allow_pickle=True)
    exist_spam_dic = np.load('./dicts/exist_spam_dict'+sys.argv[1]+'_'+str(int(sys.argv[2])-1)+'.npy',allow_pickle=True)
    stopwords = []
    stopwordsfile = open('./stopwords.txt','r')
    for line in stopwordsfile:
        line = line.strip()
        stopwords.append(line)
    if len(sys.argv)!=3:
        print('usage: test.py start_test_dir end_test_dir')
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
            dict_of_words['number_in_content']=0
            dict_of_words['url_in_content']=0
            dict_of_words['email_in_content']=0
            dict_of_words['number_in_title']=0
            dict_of_words['url_in_title']=0
            dict_of_words['email_in_title']=0
            dict_of_words['label_of_the_email']=labels[path]
            content = False 
            title = False
            try:
                for line in file:
                    line = line.strip()
                    if line == '':
                        content = True
                    words = line.split()
                    if title:
                        title = False
                    if len(words)>0 and words[0] == 'Subject:' and not content:
                        title = True
                        words.pop(0)                    
                    punctuation = [',','.','?','!','<','>','"',"'",'(',')','-',':',';','&','~','^','[',']','{','}','|','/','*','#','=','_','+','%']
                    for word in words:
                        if content:
                            word,add_num = eliminate_num_from_str(word)
                            dict_of_words['number_in_content'] += add_num
                            while word != '' and word[0] in punctuation:
                                word = word.strip(word[0])
                            while word != '' and word[-1] in punctuation:
                                word = word.strip(word[-1])
                            if word != '':
                                if validators.url(word):
                                    dict_of_words['url_in_content'] += 1
                                elif validators.email(word):
                                    dict_of_words['email_in_content'] += 1
                                else:
                                    if word.lower() not in stopwords:
                                        word = word+'_content'
                                        # print(word)
                                        if word in dict_of_words.keys():
                                            dict_of_words[word]+=1
                                        else:
                                            dict_of_words[word]=1
                        elif title:
                            word,add_num = eliminate_num_from_str(word)
                            dict_of_words['number_in_title'] += add_num
                            if word != '':
                                if validators.url(word):
                                    dict_of_words['url_in_title'] += 1
                                elif validators.email(word):
                                    dict_of_words['email_in_title'] += 1
                                else:
                                    if word.lower() not in stopwords:
                                        word = word+'_title'
                                        if word in dict_of_words.keys():
                                            dict_of_words[word]+=1
                                        else:
                                            dict_of_words[word]=1
                # for key in dict_of_words.keys():
                #     print(key)
                # py
                # 
                # print(dict_of_words)
                # if not os.path.exists('./dicts/'+dirname):
                #     os.makedirs('./dicts/'+dirname) 
                # np.save('./dicts/'+dirname+'/'+filename,dict_of_words)
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
                alpha = 0.00001
                for key in dict_of_words.keys():
                    if key!='label_of_the_email':
                        if key in exist_spam_dict.keys() and key in train_data.keys():
                            p_x_spam*=(exist_spam_dict[key]+alpha)/(exist_dict['label_of_the_email']+alpha*exist_dict['count'])
                            p_x_ham*=(exist_dict[key] - exist_spam_dict[key]+alpha)/(exist_dict['count']-exist_dict['label_of_the_email']+alpha*exist_dict['count'])
                        else:
                            p_x_spam*=alpha/(exist_dict['label_of_the_email']+alpha*exist_dict['count'])
                            p_x_ham*=alpha/(exist_dict['count']-exist_dict['label_of_the_email']+alpha*exist_dict['count'])
                p_is_spam = p_spam*p_x_spam/(p_spam*p_x_spam+p_x_ham*p_ham)
                # print(p_is_spam)
                # print(path)
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
        print('false positive rate: '+str(predict_spam_wrong/ham_total))
        print('false negative rate: '+str(predict_ham_wrong/spam_total))
if __name__ == "__main__":
   main(sys.argv[1:])
