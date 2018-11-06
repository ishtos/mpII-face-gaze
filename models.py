import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GazeNet(nn.Module):

    def __init__(self):
        super(GazeNet, self).__init__()
        model = models.alexnet(pretrained=True)
        self.alexnet = model.features

        self.Conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.Conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.Conv3 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.layer1 = nn.Sequential(
            nn.Linear(256*13*13, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )   
        self.fc = nn.Linear(4096, 2)

        self._initialize_weight()
        self._initialize_bias()

    def _initialize_weight(self):
        nn.init.normal_(self.Conv1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.Conv2.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.Conv3.weight, mean=0.0, std=0.001)

    def _initialize_bias(self):
        nn.init.constant_(self.Conv1.bias, val=0.1)
        nn.init.constant_(self.Conv2.bias, val=0.1)
        nn.init.constant_(self.Conv3.bias, val=1)

    def forward(self, x, z):
        x = self.alexnet(x)

        y = F.relu(self.Conv1(x))
        y = F.relu(self.Conv2(y))
        y = F.relu(self.Conv3(y))

        x = F.dropout(F.relu(F.mul(x, y)), 0.5)
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)

        return x