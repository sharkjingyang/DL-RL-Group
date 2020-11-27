import numpy as np 
import torch 
import matplotlib.pyplot as plt
import torch.nn as nn
import argparse

# make data set
device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
cvt = lambda x: x.to(device, non_blocking=True)

x=torch.linspace(-5.0,5.0,1000).reshape(-1,1)
x=cvt(x)
y=torch.pow(x,3)-3*torch.pow(x,2)+torch.randn(x.size()).to(device)


parser = argparse.ArgumentParser('practice')
parser.add_argument("--nres",type=int,default=1,help="num of resnet layers")
parser.add_argument("--width",type=int,default=10,help="width of resnet layers")
parser.add_argument("--lr",type=int,default=0.05,help="learning rate")
args = parser.parse_args()



class ResNet(nn.Module):
    def __init__(self,nres,width):
        super(ResNet,self).__init__()
        
        self.nres=nres
        self.h=1/self.nres
        self.width=width
        self.act=nn.ReLU(True)
        self.layers=nn.ModuleList([])
        self.layers.append(nn.Linear(1,self.width,bias=True))      
        for i in range(self.nres):
            self.layers.append(nn.Linear(self.width,self.width,bias=True))
        self.layers.append(nn.Linear(self.width,1,bias=True))  
        

    def forward(self,x):
      
        x= self.act(self.layers[0].forward(x))
        for i in range(1,self.nres):
            x= x+self.act(self.layers[i].forward(x))
        x= self.layers[self.nres+1].forward(x)
        # x= self.act(self.layers[2].forward(x))
 
        return  x






if __name__ == '__main__':
    nres=args.nres
    width=args.width
    net=ResNet(nres=nres,width=width)
    net=net.to(device)

    Loss=nn.MSELoss()
    optimizer =torch.optim.Adam(net.parameters(),lr=0.1)


    for t in range(1000):
    
        prediction=net(x)
        loss=Loss(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("iter " +str(t) +"is loss is "+ str(loss))
        print(net.layers[0].weight.reshape(1,10))

        # if t %10 ==0:
        #     plt.cla()
        #     plt.scatter(x.cpu(), y.cpu(),s=2)
        #     plt.plot(x.cpu().numpy(), prediction.data.cpu().numpy(), 'r-', lw=2)
        #     plt.text(0.5, 0, 'Loss = %f  t = %f' %( loss.item() ,t))
        #     plt.pause(0.2)

