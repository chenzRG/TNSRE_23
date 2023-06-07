import torch
import numpy as np
from sklearn.model_selection import train_test_split

class Mydatasets(torch.utils.data.Dataset):
    def __init__(self, data,label ,transform = None):
        self.transform = transform

        self.data = data
        self.label = label

        self.datanum = len(data)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        
        out_data = torch.tensor(self.data[idx]).float()   
        #out_data = torch.tensor(self.data[idx]).short()
        #out_data = torch.tensor(self.data[idx]).double() 
        
        #out_data = out_data.type(torch.HalfTensor)  #16-bit floating point
        #out_data = out_data.type(torch.FloatTensor)  #32-bit floating point
        #out_data = out_data.type(torch.DoubleTensor)  #64-bit floating point
        out_label = torch.tensor(self.label[idx])
        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label
    
# Data loading and processing 
def load_data(dat1, dat2, label):
    ###Frequency band segmentation###### 
    dat = np.concatenate((dat1.reshape(-1,1,32,30), dat2.reshape(-1,1,32,30)), axis=1)      
    fixdata = dat[:,:,0:16,:]  
    mean_p1 = np.mean(dat[:,:,16:20,:], axis = 2)
    mean_p2 = np.mean(dat[:,:,20:24,:], axis = 2)
    mean_p3 = np.mean(dat[:,:,24:28,:], axis = 2)
    mean_p4 = np.mean(dat[:,:,28:32,:], axis = 2)
    num_data = len(dat)
    ch = 2
    inputdat = np.concatenate((fixdata,mean_p1.reshape(num_data, ch, 1, 30),mean_p2.reshape(num_data, ch, 1, 30),mean_p3.reshape(num_data, ch, 1, 30),mean_p4.reshape(num_data, ch, 1, 30)),axis=2)
    print("input data shape:", inputdat.shape)
    ###

    train, test, train_label, test_label = train_test_split(inputdat, np.array(label), test_size = 0.1, stratify = label, random_state = 0)
    print('train data:',len(train))
    print('test data:',len(test))

    train_data_set = Mydatasets(data = train,label = train_label)
    test_data_set = Mydatasets(data = test,label = test_label)

    return train_data_set, test_data_set
