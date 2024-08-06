import os
import torch
import numpy as np

def save_npy_record(npy_path, record, name=None):
    if name == None:
        name = "record"
    max_index = 0
    for filename in os.listdir(npy_path):
        if filename.startswith(name) and filename.endswith(".npy"):
            max_index +=1
    if max_index==0:
        np.save(npy_path+'/{}.npy'.format(name), record)
    else:
        np.save(npy_path+'/{}_{}.npy'.format(name, max_index), record)

def get_device(gpu):
    device= torch.device('cpu')
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:'+str(gpu))
    return device

if __name__ == "__main__":
    print("done")
