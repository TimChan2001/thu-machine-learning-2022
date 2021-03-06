import os, sys, validators, getopt
import numpy as np

"""
eliminate_num_from_str方法从字符串中去除数字提取内容
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
"""
邮件中bag of words特征的提取
"""
def main(argv): 
    stopwords = []
    exist_dict = {}
    exist_spam_dict = {}
    exist_dict['count']=0
    stopwordsfile = open('./stopwords.txt','r') # 从stopwords.txt中parse出停用词
    for line in stopwordsfile:
        line = line.strip()
        stopwords.append(line)
    if len(sys.argv)!=3:
        print('usage: parse_email.py start_test_dir end_test_dir')
        sys.exit()
    label = open('./label/index','r') # parse出label
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
        if i not in range(int(sys.argv[1]),int(sys.argv[2])): # 遍历选中作为train set的邮件
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
                dict_of_words = {} # 本邮件bag of words对应的字典
                dict_of_words['number_in_content']=0 # 内容中的数字（包括带数字的词）
                dict_of_words['url_in_content']=0 # 内容中的链接
                dict_of_words['email_in_content']=0 # 内容中的电子邮件信箱
                dict_of_words['number_in_title']=0 # 标题中的数字（包括带数字的词）
                dict_of_words['url_in_title']=0 # 标题中的链接
                dict_of_words['email_in_title']=0 # 标题中的电子邮件信箱
                dict_of_words['label_of_the_email']=labels[path] # 邮件的label
                content = False 
                title = False
                try:
                    for line in file:
                        line = line.strip()
                        if line == '': # 观察特点发现邮件头和内容之间有一个空行，用于区分头和内容
                            content = True
                        words = line.split()
                        if title:
                            title = False
                        if len(words)>0 and words[0] == 'Subject:' and not content:
                            title = True
                            words.pop(0)                    
                        punctuation = [',','.','?','!','<','>','"',"'",'(',')','-',':',';','&','~','^','[',']','{','}','|','/','*','#','=','_','+','%']
                        for word in words:
                            if content: # content中的词处理
                                word,add_num = eliminate_num_from_str(word)
                                dict_of_words['number_in_content'] += add_num
                                while word != '' and word[0] in punctuation: # 去掉首尾的无关符号
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
                                            word = word+'_content' # 在content中的词加上“_content”和title中的词区分开
                                            if word in dict_of_words.keys():
                                                dict_of_words[word]+=1
                                            else:
                                                dict_of_words[word]=1
                            elif title: # title中的词处理
                                word,add_num = eliminate_num_from_str(word)
                                dict_of_words['number_in_title'] += add_num
                                if word != '':
                                    if validators.url(word):
                                        dict_of_words['url_in_title'] += 1
                                    elif validators.email(word):
                                        dict_of_words['email_in_title'] += 1
                                    else:
                                        if word.lower() not in stopwords:
                                            word = word+'_title' # 在title中的词加上“_title”和content中的词区分开
                                            if word in dict_of_words.keys():
                                                dict_of_words[word]+=1
                                            else:
                                                dict_of_words[word]=1
                    for key in dict_of_words.keys(): # 把单邮件中的bag of words同步到全局dicts中
                        if dict_of_words[key]>0:
                            if key in exist_dict.keys():
                                exist_dict[key]+=1
                            else:
                                exist_dict[key]=1
                    if dict_of_words['label_of_the_email'] == 1: # spam文件
                        for key in dict_of_words.keys(): # 把单垃圾邮件中的bag of words同步到全局垃圾dicts中
                            if dict_of_words[key]>0:
                                if key in exist_spam_dict.keys():
                                    exist_spam_dict[key]+=1
                                else:
                                    exist_spam_dict[key]=1
                    exist_dict['count']+=1
                    file.close()
                except:
                    continue
    np.save('./dicts/exist_dict'+sys.argv[1]+'_'+str(int(sys.argv[2])-1)+'.npy',exist_dict) # 存
    np.save('./dicts/exist_spam_dict'+sys.argv[1]+'_'+str(int(sys.argv[2])-1)+'.npy',exist_spam_dict)

if __name__ == "__main__":
   main(sys.argv[1:])
