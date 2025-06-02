import torch
import torchvision
import torch.utils.data as Data
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import numpy as np

starttime = time.time()

torch.manual_seed(1)#随机种子

# 超参数
EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
#DOWNLOAD_MNIST = True   # 下过数据的话, 就可以设置成 False
N_TEST_IMG = 5          # 到时候显示 5张图片看效果


# Mnist digits dataset
train_data = torchvision.datasets.MNIST(
    root='./data/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=True,                        # download it if you don't have it
)

loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# # plot one example
# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# # print(train_data.train_data[0])
# plt.imshow(train_data.train_data[2].numpy(),cmap='Greys')
# plt.title('%i'%train_data.train_labels[2])
# plt.show()


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
         # 压缩
        self.encoder  =  nn.Sequential(
            nn.Linear(28*28,128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 3)
        )
        # 解压
        self.decoder = nn.Sequential(
            nn.Linear(3,16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()  # 激励函数让输出值在 (0, 1)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

#training
autoencoder = AutoEncoder()
print(autoencoder)

optimizer = torch.optim.Adam(autoencoder.parameters(),lr=LR)
loss_func = nn.MSELoss()

# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))  #return fig, ax
plt.ion()   # continuously plot

# original data (first row) for viewing
view_data = train_data.data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255

for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())


for epoch in range(EPOCH):
    for step,(x,y) in enumerate(loader):
        b_x = x.view(-1,28*28)  # batch x, shape (batch, 28*28)
        b_y = x.view(-1,28*28)  # batch x, shape (batch, 28*28)
        decoded = autoencoder(b_x)

        loss = loss_func(decoded, b_y)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        if step%50 == 0:
            print('Epoch :', epoch,'|','train_loss:%.4f'%loss.data)
            # plotting decoded image (second row)
            decoded_data = autoencoder(view_data)
            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(()); a[1][i].set_yticks(())
            plt.draw(); plt.pause(0.05)
plt.ioff()

torch.save(autoencoder,'AutoEncoder.pkl')
print('________________________________________')
print('finish training')

endtime = time.time()
print('训练耗时：',(endtime - starttime))


