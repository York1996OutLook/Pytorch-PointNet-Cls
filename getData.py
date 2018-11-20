import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
main_path="E:/DataSets/shapenet_part_seg_hdf5_data/hdf5_data/"
train_txt_path=main_path+"train_hdf5_file_list.txt"
valid_txt_path=main_path+"val_hdf5_file_list.txt"

def get_data(train=True):
    data_txt_path =train_txt_path if train else valid_txt_path

    with open(data_txt_path, "r") as f:
        txt = f.read()
    clouds_li = []
    labels_li = []
    for file_name in txt.split():
        h5 = h5py.File(main_path + file_name)
        pts = h5["data"].value
        lbl = h5["label"].value
        clouds_li.append(torch.Tensor(pts))
        labels_li.append(torch.Tensor(lbl))
    clouds = torch.cat(clouds_li)
    labels = torch.cat(labels_li)
    return clouds,labels.long().squeeze()

class PointDataSet(Dataset):
    def __init__(self,train=True):

        clouds, labels = get_data(train=train)

        self.x_data=clouds
        self.y_data=labels

        self.lenth=clouds.size(0)
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.lenth

def get_dataLoader(train=True):
    point_data_set=PointDataSet(train=train)
    data_loader=DataLoader(dataset=point_data_set,batch_size=16,shuffle=train)
    return data_loader