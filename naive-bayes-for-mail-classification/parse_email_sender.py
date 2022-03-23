import os, sys, validators, getopt
import numpy as np
"""
邮件中sender email addr特征的提取
"""
def main(argv): # 代码注释见同目录下的parse_email.py，逻辑相同
    exist_sender_dict = {}
    exist_sender_spam_dict = {}
    exist_sender_dict['count']=0
    if len(sys.argv)!=3:
        print('usage: parse_email_sender.py start_test_dir end_test_dir')
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
    for i in range(127):
        if i not in range(int(sys.argv[1]),int(sys.argv[2])):
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
                    for key in dict_of_words.keys():
                        if dict_of_words[key]>0:
                            if key in exist_sender_dict.keys():
                                exist_sender_dict[key]+=1
                            else:
                                exist_sender_dict[key]=1
                    if dict_of_words['label_of_the_email'] == 1:#spam文件
                        for key in dict_of_words.keys():
                            if dict_of_words[key]>0:
                                if key in exist_sender_spam_dict.keys():
                                    exist_sender_spam_dict[key]+=1
                                else:
                                    exist_sender_spam_dict[key]=1
                    exist_sender_dict['count']+=1
                    file.close()
                except:
                    continue
    np.save('./dicts/exist_sender_dict'+sys.argv[1]+'_'+str(int(sys.argv[2])-1)+'.npy',exist_sender_dict)
    np.save('./dicts/exist_sender_spam_dict'+sys.argv[1]+'_'+str(int(sys.argv[2])-1)+'.npy',exist_sender_spam_dict)
if __name__ == "__main__":
   main(sys.argv[1:])
