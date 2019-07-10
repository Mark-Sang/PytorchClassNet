import torch
import torch.nn
import torch.optim

x = torch.tensor([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]).cuda()
y = torch.tensor([[.3], [.5], [.7]]).cuda()

class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1)                       
            )
        self.opt = torch.optim.Adam(self.parameters(), lr=0.01)
        self.mls = torch.nn.MSELoss()

    def forward(self, input):
        return self.fc(input)

    def train_model(self, x, y):
        out = self.forward(x)
        loss = self.mls(out, y).cuda()
        print(loss)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def test(self, x):
        return self.forward(x)

net = MyNet().cuda()
for i in range(10000):
    net.train_model(x,y)

out = net.test(x).cuda()
print(out)

k = torch.tensor([[0.1, 0.8]]).cuda()
j = net.test(k).cuda()
print("j=", j)

