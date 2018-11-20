import torch
import torch.nn as nn
import getData
import datetime
class PointNet(nn.Module):
    def __init__(self,point_num):

        super(PointNet, self).__init__()

        self.inputTransform=nn.Sequential(
            nn.Conv2d(1,64,(1,3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 1024,1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.MaxPool2d((point_num,1)),
        )
        self.inputFC = nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256,9),
        )
        self.mlp1=nn.Sequential(
            nn.Conv2d(1,64,(1,3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64,64,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.featureTransform = nn.Sequential(
            nn.Conv2d(64, 64,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 1024,1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.MaxPool2d((point_num, 1)),
        )
        self.featureFC=nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64*64),
        )
        self.mlp2=nn.Sequential(
            nn.Conv2d(64,64,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64,128,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.fc=nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(p=0.7,inplace=True),
            nn.Linear(256,16),
            nn.Softmax(dim=1),
        )
        self.inputFC[4].weight.data=torch.zeros(3*3,256)
        self.inputFC[4].bias.data=torch.eye(3).view(-1)
    def forward(self, x):               #[B, N, XYZ]
        '''
            B:batch_size
            N:point_num
            K:k_classes
            XYZ:input_features
        '''
        batch_size=x.size(0)#batchsize大小
        x=x.unsqueeze(1)                #[B, 1, N, XYZ]

        t_net=self.inputTransform(x)    #[B, 1024, 1,1]
        t_net=t_net.squeeze()           #[B, 1024]
        t_net=self.inputFC(t_net)       #[B, 3*3]
        t_net=t_net.view(batch_size,3,3)#[B, 3, 3]

        x=x.squeeze()                   #[B, N, XYZ]

        x=torch.stack([x_item.mm(t_item) for x_item,t_item in zip(x,t_net)])#[B, N, XYZ]# 因为mm只能二维矩阵之间，故逐个乘再拼起来

        x=x.unsqueeze(1)                #[B, 1, N, XYZ]

        x=self.mlp1(x)                  #[B, 64, N, 1]

        t_net=self.featureTransform(x)  #[B, 1024, 1, 1]
        t_net=t_net.squeeze()           #[B, 1024]
        t_net=self.featureFC(t_net)     #[B, 64*64]
        t_net=t_net.view(batch_size,64,64)#[B, 64, 64]

        x=x.squeeze().permute(0,2,1)    #[B, N, 64]

        x=torch.stack([x_item.mm(t_item)for x_item,t_item in zip(x,t_net)])#[B, N, 64]

        x=x.permute(0,2,1).unsqueeze(-1)#[B, 64, N, 1]

        x=self.mlp2(x)                  #[B, N, 64]

        x,_=torch.max(x,2)              #[B, 1024, 1]

        x=self.fc(x.squeeze())          #[B, K]
        return x

EPOCHES=100
POINT_NUM=2048

train_loader=getData.get_dataLoader(train=True)
test_loader=getData.get_dataLoader(train=False)

net=PointNet(POINT_NUM).cuda()

optimizer=torch.optim.Adam(net.parameters(),weight_decay=0.001)
loss_function=nn.CrossEntropyLoss()

for epoch in range(EPOCHES):
    time_start=datetime.datetime.now()
    net.train()
    for cloud,label in train_loader:
        cloud,label=cloud.cuda(),label.cuda()
        out = net(cloud)
        loss=loss_function(out,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    total=0
    net.eval()
    for cloud,label in test_loader:
        cloud,label=cloud.cuda(),label.cuda()
        out=net(cloud)
        _,pre=torch.max(out,1)
        correct=(pre==label).sum()
        total+=correct.item()
    time_end=datetime.datetime.now()
    time_span_str=str((time_end-time_start).seconds)
    print(str(epoch+1)+"迭代期准确率："+ str(total/len(test_loader.dataset))+"耗时"+time_span_str+"S")

#python的强大之处
acc=sum([(torch.max(net(cloud.cuda()),1)[1]==label.cuda()).sum() for cloud,label in test_loader]).item()/len(test_loader.dataset)