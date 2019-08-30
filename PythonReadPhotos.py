from dataloader import TrainPhotos,TestPhotos
from Net import discriminator,generator          #testphototansform,trainphototansform
from torchvision.utils import save_image
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch

def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out

TrainData=TrainPhotos('./train')
TestData=TestPhotos('./test')

if __name__ == '__main__':
  #  TrainData=TrainPhotos('./train')
  #  TestData=TestPhotos('./test')
 #   print(TrainData[2],'\n')
 #   print(TestData.__getitem__(2).size())  
    criterion = nn.BCELoss()
    num_img = 1
    #z_dimension = 784
    D = discriminator().cuda()
    G = generator().cuda()
   # G = generator(z_dimension, 3136).cuda()
    #test = testphototansform().cuda()
    #train = trainphototansform().cuda()
    d_optimizer = optim.Adam(D.parameters(), lr = 0.0003)
    g_optimizer = optim.Adam(G.parameters(), lr = 0.0003)

    count = 0
    epoch = 5
    gepoch = 1
    numm = 0
    print("start")
    for k in range(500):
        for i in range(3):
            testdata = TestData[i].cuda()
            traindata = TrainData[i].cuda()
            testdata = testdata.unsqueeze(0).cuda()
            traindata = traindata.unsqueeze(0).cuda()

            #Test = test(testdata).cuda()
            #Train = train(traindata).cuda()

            real_label = Variable(torch.ones(num_img)).cuda()
            fake_label = Variable(torch.zeros(num_img)).cuda()

            real_out = D(testdata)
            d_loss_real = criterion(real_out, real_label)
            real_score =real_out

            # z = Variable(torch.randn(num_img, z_dimension)).cuda()
            fake_img = G(traindata)
            fake_out = D(fake_img.detach()) #detach()将D和G网络隔开
            d_loss_fake = criterion(fake_out, fake_label)
            fake_score = fake_out
        
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            fake_label = Variable(torch.ones(num_img)).cuda()
            z = traindata
            fake_img = G(z)
            output = D(fake_img)
            g_loss = criterion(output, fake_label)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} D real: {:.6f}, D fake: {:.6f}'.format(
                k, 100, d_loss.data, g_loss.data,
                real_score.data.mean(), fake_score.data.mean()))
        
        #    fake_images = to_img(fake_img.cpu().data)
            fake_images = fake_img
            save_image(fake_images, './dc_img/fake_images-{}.png'.format(numm))
            numm = numm+1
    #        plt.show()
    #        count = count + 1

a =  TrainData[3].cuda()
a = a.unsqueeze(0).cuda()
aa = G(a)
save_image(aa, './dc_img/fake_images-{}.png'.format(100000))