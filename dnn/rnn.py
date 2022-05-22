from sklearn import svm
import numpy as npy
from gensim.models import Word2Vec
import datetime
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import mean_absolute_error,mean_squared_error

model = Word2Vec.load("word2vec.model")
X_train = npy.load('./X_train_20000.npy',allow_pickle=True).tolist()
X_test = npy.load('./X_test_2000.npy',allow_pickle=True).tolist()
train_idx = []
test_idx = []

H = 256
LAYERS = 5
DROPOUT = 0.1
BATCH_SIZE = 32
EPOCHS = 10
out = open('rnn_'+str(H)+'h_'+str(LAYERS)+'l_epoch_0.log','w')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

stopwords = []
stopwordsfile = open('./stopwords.txt','r') # 从stopwords.txt中parse出停用词
for line in stopwordsfile:
    line = line.strip()
    stopwords.append(line)

def text2vec(text):
    sentence = []
    complement_vec = []
    for i in range(50):
        complement_vec.append(0)
    idx = 0
    while len(sentence)<128:
        try:
            sentence.append(model.wv[text[idx]].tolist())
            idx+=1
        except:
            idx+=1
        if idx >= len(text):
            break
    while len(sentence)<128:
        sentence.append(complement_vec)
    return sentence

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=50, hidden_size=H,
                            num_layers=LAYERS, batch_first=True, dropout=DROPOUT)
        self.fullconnect = nn.Sequential(
            nn.ReLU(),
            nn.Linear(H * 128, 5)
        )

    def forward(self, inputs):
        states, hidden = self.lstm(inputs)
        outputs = self.fullconnect(
        states.reshape(inputs.shape[0], H * 128))
        return outputs

def initialize():
    rnn = RNN()
    rnn.to(device)
    optimizer = torch.optim.Adam(rnn.parameters(), weight_decay=1e-8)
    loss_fn = nn.CrossEntropyLoss()
    return rnn, optimizer, loss_fn

def train(model, optimizer, loss_fn, train_dataloader):
    train_loss = 0
    start_time = datetime.datetime.now().timestamp()
    for step, batch in enumerate(train_dataloader):
        inputs, labels = tuple(t.to(device) for t in batch)
        model.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        train_loss+=loss.item()
        loss.backward()
        optimizer.step()
        if (step % 60 == 0 and step != 0) or (step == len(train_dataloader) - 1):
            end_time = datetime.datetime.now().timestamp()
            print('progress: '+str(round(100*(step+1)/len(train_dataloader)))+'%   train loss: '+str(train_loss/(step+1))+'   train time: '+str(round(end_time-start_time))+'s')
            out.write('progress: '+str(round(100*(step+1)/len(train_dataloader)))+'%   train loss: '+str(train_loss/(step+1))+'   train time: '+str(round(end_time-start_time))+'s\n')

def main():
    global out
    # for item in X_train:
    #     train_idx.append(item[2])
    # for item in X_test:
    #     test_idx.append(item[2])
    # test_vec = []
    # test_label = []
    # train_vec = []
    # train_label = []
    # items = open('/root/thu-machine-learning-2022/ensemble-learning/exp3-reviews.csv','r')
    # idx = 0
    # for line in items:
    #     line = line.strip()
    #     print(idx)
    #     if (line.split('\t'))[0] != 'overall':
    #         train = False
    #         test = False
    #         text = []
    #         if idx in train_idx:
    #             train = True
    #         if idx in test_idx:
    #             test = True
    #         idx+=1
    #         if idx == 22001:
    #             break
    #         rating = (line.split('\t'))[0]
    #         if train:
    #             train_label.append(int(rating[0])-1)
    #         if test:
    #             test_label.append(int(rating[0])-1)
    #         summary = (line.split('\t'))[-2]
    #         reviewText = (line.split('\t'))[-1]
    #         end_punctuation = ['.','?','!','~']
    #         mid_punctuation = [',','<','>','"','(',')',':',';','&','^','[',']','{','}','|','/','*','#','=','_','+','%',"'"]
    #         sentence = summary.split()
    #         for i in range(len(sentence)-1,-1,-1):
    #             while len(sentence[i])>0 and sentence[i][-1] in mid_punctuation:
    #                 sentence[i] = sentence[i].strip(sentence[i][-1])
    #             while len(sentence[i])>0 and sentence[i][0] in mid_punctuation: 
    #                 sentence[i] = sentence[i].strip(sentence[i][0]) 
    #             while len(sentence[i])>0 and sentence[i][-1] in end_punctuation:
    #                 sentence[i] = sentence[i].strip(sentence[i][-1])
    #             while len(sentence[i])>0 and sentence[i][0] in end_punctuation: 
    #                 sentence[i] = sentence[i].strip(sentence[i][0])     
    #             if sentence[i].isdigit() or sentence[i].lower() in stopwords or sentence[i]=='':
    #                 del sentence[i]
    #         if len(sentence) > 1 :
    #             for word in sentence:
    #                 text.append(word)
    #         start = 0
    #         for end in range(len(reviewText)):
    #             if reviewText[end] in end_punctuation:
    #                 sentence = reviewText[start:end].split()
    #                 start = end+1
    #                 for i in range(len(sentence)-1,-1,-1):
    #                     while len(sentence[i])>0 and sentence[i][-1] in mid_punctuation:
    #                         sentence[i] = sentence[i].strip(sentence[i][-1])
    #                     while len(sentence[i])>0 and sentence[i][0] in mid_punctuation: 
    #                         sentence[i] = sentence[i].strip(sentence[i][0])   
    #                     if sentence[i].isdigit() or sentence[i].lower() in stopwords or sentence[i]=='':
    #                         del sentence[i]
    #                 if len(sentence) > 1:
    #                     for word in sentence:
    #                         text.append(word)
    #         vec = text2vec(text)
    #         if train:
    #             train_vec.append(vec)
    #         if test:
    #             test_vec.append(vec)
    # npy.save('train_vec.npy',train_vec)
    # npy.save('test_vec.npy',test_vec)
    # npy.save('train_label.npy',train_label)
    # npy.save('test_label.npy',test_label)
    train_vec = npy.load('./train_vec.npy',allow_pickle=True).tolist()
    test_vec = npy.load('./test_vec.npy',allow_pickle=True).tolist()
    train_label = npy.load('./train_label.npy',allow_pickle=True).tolist()
    test_label = npy.load('./test_label.npy',allow_pickle=True).tolist()
    print("size of train set: "+str(len(train_vec)))
    print("size of test set: "+str(len(test_vec)))
    train_data = Data.TensorDataset(torch.tensor(train_vec), torch.tensor(train_label))
    train_dataloader = Data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_data = Data.TensorDataset(torch.tensor(test_vec), torch.tensor(test_label))
    test_dataloader = Data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    rnn, optimizer, loss_fn = initialize()
    for epoch_i in range(EPOCHS):
        print('epoch '+str(epoch_i)+' ......')
        out.write('epoch '+str(epoch_i)+' ......\n')
        train(rnn, optimizer, loss_fn, train_dataloader)
        test_loss = []
        test_accuracy = []
        test_mae = []
        test_rmse = []
        for batch in test_dataloader:
            input_ids, labels = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                logits = rnn(input_ids)  
            loss = loss_fn(logits, labels)
            test_loss.append(loss.item())
            preds = torch.argmax(logits, dim=1).flatten()
            accuracy = (preds == labels).cpu().numpy().mean() * 100
            mae = mean_absolute_error(preds,labels)
            rmse = npy.sqrt(mean_squared_error(preds,labels))
            test_accuracy.append(accuracy)
            test_mae.append(mae)
            test_rmse.append(rmse)
        val_loss = npy.mean(test_loss)
        val_accuracy = npy.mean(test_accuracy)
        val_mae = npy.mean(test_mae)
        val_rmse = npy.mean(test_rmse)
        print('test loss: '+str(val_loss)+'   acc: '+str(val_accuracy)+'%   MAE: '+str(val_mae)+'   RMSE: '+str(val_rmse))
        out.write('test loss: '+str(val_loss)+'   acc: '+str(val_accuracy)+'%   MAE: '+str(val_mae)+'   RMSE: '+str(val_rmse)+'\n')
        out.close()
        if epoch_i < EPOCHS-1:
            out = open('rnn_'+str(H)+'h_'+str(LAYERS)+'l_epoch_'+str(epoch_i+1)+'.log','w')

if __name__ == "__main__":
   main()