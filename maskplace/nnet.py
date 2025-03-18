import torch.nn as nn
import torch


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
        )
    def forward(self, x):
        return self.cnn(x)


class MyCNNCoarse(nn.Module):
    def __init__(self, res_net, device):
        super(MyCNNCoarse, self).__init__()
        self.cnn = res_net.to(device)
        self.cnn.fc = torch.nn.Linear(512, 16*7*7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding = 1), #14
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding = 1), #28
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, 3, stride=2, padding=1, output_padding = 1), #56
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, 3, stride=2, padding=1, output_padding = 1), #112
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding = 1), #224
        )
    def forward(self, x):
        x = self.cnn(x).reshape(-1, 16, 7, 7)
        return self.deconv(x)
