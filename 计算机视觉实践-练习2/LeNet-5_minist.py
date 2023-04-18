import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
batch_size = 64

train_dataset = datasets.MNIST(root='D:\pybook\dataset\minist/',
                            train=True,
                            transform=transform,download= True)
train_loader = DataLoader(dataset=train_dataset,shuffle= True,batch_size=batch_size,num_workers=4)

test_dataset = datasets.MNIST(root='D:\pybook\dataset\minist/',
                            train=False,
                            transform=transform,download= True)
test_loader = DataLoader(dataset=test_dataset,shuffle=False,batch_size=batch_size,num_workers=4)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.AvgPool2d(kernel_size=2),
            nn.Sigmoid()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.AvgPool2d(kernel_size=2),
            nn.Sigmoid()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=4 * 4 * 16, out_features=120)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, input):
        conv1_output = self.conv1(input)  # [28,28,1]-->[24,24,6]-->[12,12,6]
        conv2_output = self.conv2(conv1_output)  # [12,12,6]-->[8,8,16]-->[4,4,16]
        conv2_output = conv2_output.view(-1, 4*4*16)  # [n,4,4,16]-->[n,4*4*16]
        fc1_output = self.fc1(conv2_output)  # [n,256]-->[n,120]
        fc2_output=self.fc2(fc1_output)  # [n,120]-->[n,84]
        fc3_output = self.fc3(fc2_output)  # [n,84]-->[n,10]
        return fc3_output

model = LeNet()
#将model中数据迁移至GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'CPU')
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum= 0.05)

# 创建空列表存储每个epoch的loss和accuracy
train_losses = []
test_accuracies = []

def train(epoch):
    running_loss = 0.0
    sum_loss = 0.0
    count = 0
    for idx, (inputs, target) in enumerate(train_loader, 0):###不用dataset！！！，enumerate是为了获得当前迭代次数idx
        # 这里的代码与之前没有区别
        #将输入输出数据转移到和model同一个GPU
        inputs,target = inputs.to(device),target.to(device)
        # 正向
        y_pred = model(inputs)
        loss = criterion(y_pred, target)
        # 反向
        optimizer.zero_grad()
        loss.backward()
        # 更新
        optimizer.step()
        count = count+1
        sum_loss += loss.item()
        running_loss += loss.item()
        if idx % 300 == 299:  # 每300次打印一次平均损失，因为idx是从0开始的，所以%299，而不是300
            print(f'epoch={epoch + 1},batch_idx={idx + 1},loss={running_loss / 300}')
            running_loss = 0.0
    return sum_loss/count

def test(epoch):
    correct = 0
    total = 0
    with torch.no_grad():
        for i,data in enumerate(test_loader,0) :#如果要加enumerate，需要前面加一个i来存储序列索引
            images,labels = data
            images,labels = images.to(device),labels.to(device)
            outputs = model(images)
            _,predicted = torch.max(outputs.data,dim=1)#dim维度1意为X列，维度0是从上到下看行
            total+=labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on test_set: %d %%'%(100*correct/total))
    return correct/total

if __name__ == '__main__':
    for epoch in range(20):
        train_loss = train(epoch)
        test_accuracy = test(epoch)
        
        # 将loss和accuracy添加到对应的列表中
        train_losses.append(train_loss)
        test_accuracies.append(test_accuracy)

    # 生成并保存loss和accuracy的折线图
    plt.plot(range(1, 21), train_losses, 'b-', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()

    plt.plot(range(1, 21), test_accuracies, 'r-', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.show()

