import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as transforms
from torchvision.transforms import ToPILImage
class ResBlock(nn.Module):
    def __init__(self,out_channel):
        super(ResBlock, self).__init__()
        self.body=nn.Sequential(nn.Conv2d(out_channel,out_channel,3,1,1),
                                nn.Conv2d(out_channel,out_channel,3,1,1),
                                nn.LeakyReLU(negative_slope=1e-2,inplace=True))
    def forward(self,x1,x2):
        x1_out=self.body(x1)
        x2_out=self.body(x2)
        return x1_out,x2_out
class ChannelAttention(nn.Module):#这个好像没有用
    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.avg=nn.AdaptiveAvgPool2d(1)#特征图长款都变成1
    def forward(self,x):
        return self.avg(x)
class ChannelAttentionBlock(nn.Module):
    def __init__(self):
        super(ChannelAttentionBlock, self).__init__()
        self.ca1=nn.AdaptiveAvgPool2d(1)#平均池化成1*1的特征图
        self.ca2=nn.AdaptiveAvgPool2d(1)#平均池化成1*1的特征图
    def forward(self,x1,x2):
        eps=1e-10
        ca1 = self.ca1(x1)
        ca2 = self.ca1(x2)
        mask1_c = torch.exp(ca1) / (torch.exp(ca2) + torch.exp(ca1) + eps)
        mask2_c = torch.exp(ca2) / (torch.exp(ca1) + torch.exp(ca2) + eps)
        x1_a = mask1_c*x1
        x2_a = mask2_c*x2
        return x1_a,x2_a

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding=3 if kernel_size==7 else 1
        self.conv=nn.Conv2d(2,1,kernel_size,padding=padding,bias=False)
        self.sigmoid=nn.Sigmoid()#这行好像不用
    def forward(self,x):
        avgout = torch.mean(x,dim=1,keepdim=True)
        maxout,_ =torch.max(x,dim=1,keepdim=True)
        x=torch.cat([avgout,maxout],dim=1)
        x=self.conv(x)
        return x
class SpatialAttentionBlock(nn.Module):
    def __init__(self):
        super(SpatialAttentionBlock, self).__init__()
        self.att1=SpatialAttention()
        self.att2=SpatialAttention()
    def forward(self,x1,x2):
        eps=1e-10
        att1=self.att1(x1)
        att2=self.att2(x2)
        mask1_s = torch.exp(att1) / (torch.exp(att1) + torch.exp(att2) + eps)
        mask2_s = torch.exp(att2) / (torch.exp(att2) + torch.exp(att1) + eps)
        x1_a=mask1_s*x1
        x2_a=mask2_s*x2
        return x1_a,x2_a

class attention(nn.Module):
    def __init__(self,out_channel):
        super(attention, self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(out_channel,out_channel,stride=1,kernel_size=3,padding=1),
                                 nn.LeakyReLU(negative_slope=1e-2,inplace=True))
        self.conv2=nn.Sequential(nn.Conv2d(out_channel,out_channel,stride=1,kernel_size=3,padding=1),
                                 nn.LeakyReLU(negative_slope=1e-2,inplace=True))
        self.att_ch=ChannelAttentionBlock()
        self.att_sp=SpatialAttentionBlock()
    def forward(self,x1,x2):
        x1=self.conv1(x1)
        x2=self.conv1(x2)
        x1_c,x2_c=self.att_ch(x1,x2)
        x1_s,x2_s=self.att_sp(x1_c,x2_c)
        x1_out=self.conv2(x1_s)
        x2_out=self.conv2(x2_s)
        return x1_out, x2_out, x1, x2, x1_s, x2_s
class UFA_model(nn.Module):
    def __init__(self,in_channel=3,out_channel=64):
        super(UFA_model, self).__init__()
        self.FEB_1=nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1),
                                nn.LeakyReLU(negative_slope=1e-2,inplace=True))
        self.FEB_23=ResBlock(out_channel)
        self.FEB_45 = ResBlock(out_channel)
        self.FEB_67 = ResBlock(out_channel)
        self.UFA = attention(out_channel)
        self.FFB = nn.Conv2d(out_channel*2,out_channel,stride=1,kernel_size=1,padding=0)
        self.ICB1=nn.Sequential(nn.Conv2d(out_channel,out_channel,stride=1,kernel_size=3,padding=1),
                                nn.LeakyReLU(negative_slope=1e-2,inplace=True))
        self.ICB2 = nn.Sequential(nn.Conv2d(out_channel, out_channel, stride=1, kernel_size=3, padding=1),
                                  nn.LeakyReLU(negative_slope=1e-2, inplace=True))
        self.ICB3 = nn.Sequential(nn.Conv2d(out_channel, out_channel, stride=1, kernel_size=3, padding=1),
                                  nn.LeakyReLU(negative_slope=1e-2, inplace=True))
        self.ICB4 = nn.Sequential(nn.Conv2d(out_channel,in_channel, stride=1, kernel_size=3, padding=1),
                                  nn.Sigmoid())
        self.sig=nn.Sigmoid()
    def forward(self,x1,x2):
        F1=self.FEB_1(x1)
        F2=self.FEB_1(x2)
        F1,F2=self.FEB_23(F1,F2)
        F1,F2=self.FEB_45(F1,F2)
        F1,F2=self.FEB_67(F1,F2)
        F1,F2,F1_ori,F2_ori,F1_att,F2_att=self.UFA(F1,F2)
        fusion=self.FFB(torch.cat([F1,F2],dim=1))
        fusion=self.ICB1(fusion)
        fusion = self.ICB2(fusion)
        fusion = self.ICB3(fusion)
        output = self.ICB4(fusion)
        output = self.sig(output)
        return output,F1_ori,F2_ori,F1_att,F2_att

if __name__ == '__main__':
    image_path1 = 'lytro-02-A.jpg'
    image_path2 = 'lytro-02-B.jpg'
    img1 = Image.open(image_path1).convert("RGB")
    img2 = Image.open(image_path2).convert("RGB")
    image_tensor1 = transforms.ToTensor()(img1)
    image_tensor2 = transforms.ToTensor()(img2)
    image_tensor1_add = image_tensor1.unsqueeze(0)
    image_tensor2_add = image_tensor2.unsqueeze(0)
    net = UFA_model()
    out,F1_ori,F2_ori,F1_att,F2_att= net(image_tensor1_add,image_tensor2_add)
    print(out.shape)
    img=torch.squeeze(out,dim=0)
    print(img.shape)
    img = ToPILImage()(img)
    img.show()

# #
#
#
#

