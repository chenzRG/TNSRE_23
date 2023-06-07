import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import tqdm
import argparse
import Transformer
import dataloader


parser = argparse.ArgumentParser(description='Model Train')
parser.add_argument('--epochs', type=int, default = 300, help = 'number of epochs to train.')
parser.add_argument('--input_time_size', type=int, default = 30, help = '30 second epoch')
parser.add_argument('--t_size', type=int, default = 1, help = 'time segmentation')
parser.add_argument('--f_size', type=int, default = 4, help = "frequency band segmentation")
parser.add_argument('--num_classes', type=int, default = 5, help = '5 sleep stages')
parser.add_argument('--dim', type=int, default = 128, help = "dim of embedding")
parser.add_argument('--layers', type=int, default = 8, help = 'number of Transformer encoders')
parser.add_argument('--h', type=int, default = 12, help = "mult-heads")
parser.add_argument('--mlp_dim', type=int, default = 256, help = 'dim of MLP layers')
parser.add_argument('--ch', type=int, default = 2, help = "channels")
parser.add_argument('--h_dim', type=int, default = 16, help = "dim of each head")
parser.add_argument('--lr', type=int, default = 1e-4, help = "learning rate")
parser.add_argument('--bz', type=int, default = 64, help = "batch size")
#parser.add_argument('--dic_name', type=str, required=True, help="dic_save_name")
args = parser.parse_args()


def Train_model(train_dataloader, DEVICE):
    loss_list = []
    los_min = 10**10
    val_loss_list = []
    ac_list = []
    running_loss = 0.0
    count=0
    for _, (inputs1, labels) in enumerate(train_dataloader, 0):
        optimizer.zero_grad()
        inputs1 = inputs1.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = Transmodel(inputs1)
        
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        optimizer.step()
        count=count+1

    # print statistics
        running_loss += loss.item()
    loss_loss=running_loss/count
    loss_list.append(loss_loss)
    print('epoch',epoch+1,':finished')
    print('train_loss:',loss_loss)
    with torch.no_grad():
        count=0
        running_loss=0.0
        pre=list()
        lab=list()
        for _, (inputs1, labels) in enumerate(test_dataloader, 0):
            inputs1=inputs1.to(DEVICE)
            labels=labels.to(DEVICE)
            outputs = Transmodel(inputs1)
            loss =criterion(outputs, labels.squeeze())
            running_loss += loss.item()
            count+=1
            _, predicted = torch.max(F.softmax(outputs).data, 1)
            predicted=predicted.to(DEVICE)
            labels=labels.to(DEVICE)
            predicted=predicted.tolist()
            labels=labels.tolist()
            pre.append(predicted)
            lab.append(labels)
        loss_loss=running_loss/count

        val_loss_list.append(loss_loss)
        pre=sum(pre,[])
        lab=sum(lab,[])
        print('val_loss:',loss_loss)
        cl = classification_report(lab, pre,output_dict=True)
        print(cl)
        ac_list.append(cl['accuracy'])
        if los_min > loss_loss:
            los_min = loss_loss
            torch.save(Transmodel.state_dict(),'/opt/home/Trans_state_x') 

        #torch.save(Transmodel.state_dict(),'xxx')

if __name__ == '__main__':

    DEVICE = torch.device("cuda:1"if torch.cuda.is_available() else "cpu")

    dat1 = np.load('/xx/Data/spec_c4_1_800_30.npy')
    dat2 = np.load('/xxx/Data/spec_c3_1_800_30.npy')
    print("C4 data shape:", dat1.shape)
    print("C3 data shape:", dat2.shape)
 
     ###Read the Lable###
    index = pd.read_csv("/xxx/Data/ann_delrecords_5class.csv", header=None)
    index = index[0 : 790121].astype(int)
    print(index.apply(pd.value_counts))
    label = index.values.tolist()  #list

    train_data_set, test_data_set = dataloader.load_data(dat1, dat2, label)

    train_dataloader = torch.utils.data.DataLoader(train_data_set, batch_size = args.bz, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data_set, batch_size = args.bz, shuffle=True)

    Transmodel = Transformer.TransModel(
    image_size = args.input_time_size, 
    time_size = args.t_size, 
    fre_size = args.f_size, 
    num_classes = args.num_classes, 
    dim = args.dim, 
    depth = args.layers,  
    heads = args.h,      
    mlp_dim = args.mlp_dim,  
    channels = args.ch,
    dim_head = args.h_dim   
    ).to(DEVICE)
    # Transmodel.load_state_dict(torch.load('/Model_1_state_7'))
    optimizer = torch.optim.AdamW(Transmodel.parameters(), lr = args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    #lable weight
    #criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    EPOCH = args.epochs
        
    for epoch in tqdm.tqdm(range(EPOCH)):
        Train_model(train_dataloader, DEVICE)

