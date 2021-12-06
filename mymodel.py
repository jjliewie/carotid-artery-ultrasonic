import torch

class CCANet(torch.nn.Module):

    def __init__(self, num_class=5):
        super(CCANet, self).__init__()

        # network layers

        self.conv1 = torch.nn.Conv2d(3, 20, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(20, 50, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear1 = torch.nn.Linear(50*32*32, 250)
        self.linear2 = torch.nn.Linear(250, num_class)
    
    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = x.flatten()
        h = self.relu(self.linear1(x))
        pred = self.linear2(h)
        return pred

if __name__ == '__main__':

    model = CCANet()
    d_input = torch.randn((1, 3, 128, 128))
    pred = model(d_input)
    print(pred)
