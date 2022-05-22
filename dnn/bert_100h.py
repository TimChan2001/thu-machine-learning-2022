import torch
import torch.nn as nn
import torch.utils.data as Data
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import datetime
import numpy as np
# H = 100
BERT_H = 768
H = 100
OUTPUT = 5
EPOCHS = 10
BATCH_SIZE = 32
out = open('bert_'+str(H)+'h_epoch_0_run_2.log','w')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

checkpoint = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(checkpoint, do_lower_case=True)

def preprocessing(sequences):
    model_inputs = tokenizer(sequences, padding="max_length", truncation=True)
    input_ids = model_inputs['input_ids']
    attention_mask = model_inputs['attention_mask']
    return input_ids,attention_mask

class MyClassifier(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.bert = BertModel.from_pretrained(checkpoint)

        self.fullconnect = nn.Sequential(
            nn.Linear(BERT_H, H),
            nn.ReLU(),
            nn.Linear(H, OUTPUT)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        bert_state = outputs[0][:, 0, :]
        logits = self.fullconnect(bert_state)
        return logits

def initialize():
    classifier = MyClassifier()
    classifier.to(device)
    optimizer = torch.optim.AdamW(classifier.parameters())
    loss_fn = nn.CrossEntropyLoss()
    return classifier, optimizer, loss_fn

def train(classifier, optimizer, loss_fn, train_dataloader):
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
    items = open('/root/thu-machine-learning-2022/ensemble-learning/exp3-reviews.csv','r')
    sequences = []
    X = []
    y = []
    idx = 0
    for line in items:
        line = line.strip()
        if (line.split('\t'))[0] != 'overall':
            rating = (line.split('\t'))[0]
            summary = (line.split('\t'))[-2]
            reviewText = (line.split('\t'))[-1]
            sequences.append(summary+reviewText)
            y.append(int(rating[0])-1)
            idx+=1
            if idx == 22000:
                break
    inputs,masks = preprocessing(sequences)
    for i in range(len(inputs)):
        X.append([inputs[i],masks[i],i])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.090909)
    X_train_array = np.array(X_train)
    X_test_array = np.array(X_test)
    y_train_array = np.array(y_train)
    y_test_array = np.array(y_test)
    np.save('X_train_'+str(len(y_train))+'.npy',X_train_array)
    np.save('X_test_'+str(len(y_test))+'.npy',X_test_array)
    np.save('y_train_'+str(len(y_train))+'.npy',y_train_array)
    np.save('y_test_'+str(len(y_test))+'.npy',y_test_array)
    # X_train = np.load("./X_train_20000.npy",allow_pickle=True).tolist()
    # X_test = np.load("./X_test_2000.npy",allow_pickle=True).tolist()
    # y_train = np.load("./y_train_20000.npy",allow_pickle=True).tolist()
    # y_test = np.load("./y_test_2000.npy",allow_pickle=True).tolist()
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
    train_dataloader = Data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_data = Data.TensorDataset(inputs_test, masks_test, y_test)
    test_dataloader = Data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    classifier, optimizer, loss_fn = initialize()
    for epoch_i in range(EPOCHS):
        print('epoch '+str(epoch_i)+' ......')
        out.write('epoch '+str(epoch_i)+' ......\n')
        train(classifier, optimizer, loss_fn, train_dataloader)
        test_loss = []
        test_accuracy = []
        test_mae = []
        for batch in test_dataloader:
            input_ids, attn_mask, labels = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                logits = classifier(input_ids, attn_mask)  
            loss = loss_fn(logits, labels)
            test_loss.append(loss.item())
            preds = torch.argmax(logits, dim=1).flatten()
            accuracy = (preds == labels).cpu().numpy().mean() * 100
            mae = mean_absolute_error(preds,labels)
            test_accuracy.append(accuracy)
            test_mae.append(mae)
        val_loss = np.mean(test_loss)
        val_accuracy = np.mean(test_accuracy)
        val_mae = np.mean(test_mae)
        print('test loss: '+str(val_loss)+'   acc: '+str(val_accuracy)+'%   MAE: '+str(val_mae))
        out.write('test loss: '+str(val_loss)+'   acc: '+str(val_accuracy)+'%   MAE: '+str(val_mae)+'\n')
        out.close()
        if epoch_i < EPOCHS-1:
            out = open('bert_'+str(H)+'h_epoch_'+str(epoch_i+1)+'_run_2.log','w')

if __name__ == "__main__":
   main()


