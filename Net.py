import torch
import torch.nn as nn

#判别器(输入3*576*720，输出1)
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(3, 2, 5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((4,4)),

            nn.Conv2d(2, 1, 5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2))
        )

        self.fc=nn.Sequential(
            nn.Linear(1 * 72 * 90, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dis(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


#生成器(输入输出均为3*572*720)
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
     #   self.fc = nn.Linear(input_size, num_feature)
     #   self.br = nn.Sequential(
    #        nn.BatchNorm2d(1),
    #        nn.ReLU(True)
     #   )
        self.samll = nn.Sequential(
            nn.Conv2d(3, 50, 3, stride=1, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(True),
            nn.MaxPool2d((4, 4)),
            
            nn.Conv2d(50, 25, 3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(25, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

        self.large = nn.Sequential(
            nn.ConvTranspose2d(1, 2, 4, 2, 1, bias=False),
     #       nn.BatchNorm2d(2),
            nn.ReLU(True),

            nn.ConvTranspose2d(2, 4, 4, 2, 1, bias=False),
     #       nn.BatchNorm2d(4),
            nn.ReLU(True),

            nn.ConvTranspose2d(4, 3, 4, 2, 1, bias=False),
      #      nn.BatchNorm2d(3),
            nn.ReLU(True),
        )
        
        self.fc=nn.Sequential(
            nn.Linear(1*72*90, 1*72*90),
            nn.ReLU(True),
        )
    def forward(self, x):
   #     x = self.samll(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), 1, 72, 90)
        x = self.large(x)
        return x

##test图像转换为1*28*28
#class testphototansform(nn.module):
#    def __init__(self):
#        super(testphototansform, self).__init__()       
#        self.aaa = nn.sequential(
#            nn.conv2d(3, 2, 3, stride=1, padding=1),
#            nn.leakyrelu(0.2, true),
#            nn.maxpool2d((4, 4)),

#            nn.conv2d(2, 1, 3, stride=1, padding=1),
#            nn.leakyrelu(0.2, true),
#            nn.maxpool2d((2, 2)),
#        ) 
        
#        self.fc=nn.sequential(
#            nn.linear(1*72*90, 784),
#        )

#    def forward(self, x):
#        x = self.aaa(x)
#        x = x.view(x.size(0), 6480)
#        x = self.fc(x)
#        x = x.view(x.size(0), 1, 28, 28)
#        return x

##train图像转换为1维
#class trainphototansform(nn.module):
#    def __init__(self):
#        super(trainphototansform, self).__init__()       
#        self.br = nn.sequential(
#            nn.conv2d(3, 2, 3, stride=1, padding=1),
#            nn.maxpool2d((4, 4)),

#            nn.conv2d(2, 1, 3, stride=1, padding=1),
#            nn.maxpool2d((2, 2)),
#        ) 
        
#        self.fc=nn.sequential(
#            nn.linear(1*72*90, 784),
#        )

#    def forward(self, x):
#        x = self.br(x)
#        x = x.view(x.size(0), -1)
#        x = self.fc(x)
#        x = x.view(x.size(0), 784)
#        return x