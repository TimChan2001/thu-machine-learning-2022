import numpy as npy
from gensim.models import Word2Vec
import datetime
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import mean_absolute_error,mean_squared_error
# 其他注释见rnn.py
model = Word2Vec.load("word2vec.model")
X_train = npy.load('./X_train_20000.npy',allow_pickle=True).tolist()
X_test = npy.load('./X_test_2000.npy',allow_pickle=True).tolist()
train_idx = []
test_idx = []

DROPOUT = 0.1 # 丢弃率
CHANNEL = 128 # 输出通道数
BATCH_SIZE = 32
EPOCHS = 10
out = open('cnn_'+str(CHANNEL)+'c_'+str(DROPOUT)+'d.log','w')

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

class CNN(nn.Module): # CNN结构，具体见report
    def __init__(self):
        super().__init__()
        self.core1 = nn.Sequential(
            nn.Conv2d(1, CHANNEL, (3, 50)),
            nn.ReLU(),
            nn.MaxPool2d((128 - 2, 1)),
            nn.Dropout(DROPOUT)
        )
        self.core2 = nn.Sequential(
            nn.Conv2d(1, CHANNEL, (4, 50)),
            nn.ReLU(),
            nn.MaxPool2d((128 - 3, 1)),
            nn.Dropout(DROPOUT)
        )
        self.core3 = nn.Sequential(
            nn.Conv2d(1, CHANNEL, (5, 50)),
            nn.ReLU(),
            nn.MaxPool2d((128 - 4, 1)),
            nn.Dropout(DROPOUT)
        )
        self.fullconnect = nn.Sequential(
            nn.ReLU(),
            nn.Linear(3 * CHANNEL, 5)
        )

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        c1 = self.core1(inputs)
        c2 = self.core2(inputs)
        c3 = self.core3(inputs)
        c = torch.cat((c1, c2, c3), 3)
        outputs = c.view(inputs.shape[0], -1)
        outputs = self.fullconnect(outputs)
        return outputs

def initialize():
    cnn = CNN()
    cnn.to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    return cnn, optimizer, loss_fn

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
        if (step % 120 == 0 and step != 0) or (step == len(train_dataloader) - 1):
            end_time = datetime.datetime.now().timestamp()
            print('progress: '+str(round(100*(step+1)/len(train_dataloader)))+'%   train loss: '+str(train_loss/(step+1))+'   train time: '+str(round(end_time-start_time))+'s')
            out.write('progress: '+str(round(100*(step+1)/len(train_dataloader)))+'%   train loss: '+str(train_loss/(step+1))+'   train time: '+str(round(end_time-start_time))+'s\n')

def main():
    global out
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
    cnn, optimizer, loss_fn = initialize()
    for epoch_i in range(EPOCHS):
        print('epoch '+str(epoch_i)+' ......')
        out.write('epoch '+str(epoch_i)+' ......\n')
        train(cnn, optimizer, loss_fn, train_dataloader)
        test_loss = []
        test_accuracy = []
        test_mae = []
        test_rmse = []
        for batch in test_dataloader:
            input_ids, labels = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                logits = cnn(input_ids)  
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

if __name__ == "__main__":
   main()