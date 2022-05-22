import torch
import torch.nn as nn
import torch.utils.data as Data
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import datetime
import numpy as np

BERT_H = 768
OUTPUT = 5
EPOCHS = 10
BATCH_SIZE = 32
out = open('bert_epoch_0.log','w')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

checkpoint = 'bert-base-uncased' # 预训练模型
tokenizer = BertTokenizer.from_pretrained(checkpoint, do_lower_case=True)

def preprocessing(sequences): # sequences是summary+textReview合起来的文本
    model_inputs = tokenizer(sequences, padding="max_length", truncation=True)
    input_ids = model_inputs['input_ids']
    attention_mask = model_inputs['attention_mask']
    return input_ids,attention_mask # 提取向量化表示和attention_mask

class MyClassifier(nn.Module): # BERT架构，详见report
    def __init__(self, ):
        super().__init__()

        self.bert = BertModel.from_pretrained(checkpoint)

        self.fullconnect = self.fullconnect = nn.Sequential(
            nn.ReLU(),
            nn.Linear(BERT_H, OUTPUT)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        bert_state = outputs[0][:, 0, :]
        logits = self.fullconnect(bert_state)
        return logits

def initialize(): # 初始化模型，优化器和损失函数
    classifier = MyClassifier()
    classifier.to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(),lr = 0.01)
    loss_fn = nn.CrossEntropyLoss()
    return classifier, optimizer, loss_fn

def train(classifier, optimizer, loss_fn, train_dataloader): # 一个epoch的训练
    train_loss = 0
    start_time = datetime.datetime.now().timestamp()
    for step, batch in enumerate(train_dataloader):
        inputs, masks, labels = tuple(t.to(device) for t in batch)
        classifier.zero_grad()
        outputs = classifier(inputs,masks)
        loss = loss_fn(outputs, labels)
        train_loss+=loss.item()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
        loss.backward()
        optimizer.step()
        if (step % 60 == 0 and step != 0) or (step == len(train_dataloader) - 1):
            end_time = datetime.datetime.now().timestamp()
            print('progress: '+str(round(100*(step+1)/len(train_dataloader)))+'%   train loss: '+str(train_loss/(step+1))+'   train time: '+str(round(end_time-start_time))+'s')
            out.write('progress: '+str(round(100*(step+1)/len(train_dataloader)))+'%   train loss: '+str(train_loss/(step+1))+'   train time: '+str(round(end_time-start_time))+'s\n')

def main():
    global out
    X_train = np.load("./X_train_20000.npy",allow_pickle=True).tolist()
    X_test = np.load("./X_test_2000.npy",allow_pickle=True).tolist()
    y_train = np.load("./y_train_20000.npy",allow_pickle=True).tolist()
    y_test = np.load("./y_test_2000.npy",allow_pickle=True).tolist()
    print('size of train set: '+str(len(y_train)))
    out.write('size of train set: '+str(len(y_train))+'\n')
    inputs_train = []
    masks_train = []
    inputs_test = []
    masks_test = []
    for i in range(len(X_train)):
        inputs_train.append(X_train[i][0])
        masks_train.append(X_train[i][1])
    for i in range(len(X_test)):
        inputs_test.append(X_test[i][0])
        masks_test.append(X_test[i][1])
    inputs_train = torch.tensor(inputs_train)
    masks_train = torch.tensor(masks_train)
    y_train = torch.tensor(y_train)
    inputs_test = torch.tensor(inputs_test)
    masks_test = torch.tensor(masks_test)
    y_test = torch.tensor(y_test)
    train_data = Data.TensorDataset(inputs_train, masks_train, y_train)
    train_dataloader = Data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True) # 打包进dataloader
    test_data = Data.TensorDataset(inputs_test, masks_test, y_test)
    test_dataloader = Data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True) # 打包进dataloader
    classifier, optimizer, loss_fn = initialize()
    for epoch_i in range(EPOCHS):
        print('epoch '+str(epoch_i)+' ......')
        out.write('epoch '+str(epoch_i)+' ......\n')
        train(classifier, optimizer, loss_fn, train_dataloader)
        test_loss = []
        test_accuracy = []
        test_mae = []
        test_rmse = []
        for batch in test_dataloader: # 每个epoch后计算测试集指标
            input_ids, attn_mask, labels = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                logits = classifier(input_ids, attn_mask)  
            loss = loss_fn(logits, labels)
            test_loss.append(loss.item())
            preds = torch.argmax(logits, dim=1).flatten()
            accuracy = (preds == labels).cpu().numpy().mean() * 100
            mae = mean_absolute_error(preds,labels)
            rmse = np.sqrt(mean_squared_error(preds,labels))
            test_accuracy.append(accuracy)
            test_mae.append(mae)
            test_rmse.append(rmse)
        val_loss = np.mean(test_loss)
        val_accuracy = np.mean(test_accuracy)
        val_mae = np.mean(test_mae)
        val_rmse = np.mean(test_rmse)
        print('test loss: '+str(val_loss)+'   acc: '+str(val_accuracy)+'%   MAE: '+str(val_mae)+'   RMSE: '+str(val_rmse))
        out.write('test loss: '+str(val_loss)+'   acc: '+str(val_accuracy)+'%   MAE: '+str(val_mae)+'   RMSE: '+str(val_rmse)+'\n')
        out.close()
        if epoch_i < EPOCHS-1:
            out = open('bert_epoch_'+str(epoch_i+1)+'.log','w')

if __name__ == "__main__":
   main()


