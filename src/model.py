import torch.nn as nn
import torch.nn.functional as F

class LeaseModel(nn.Module):
    def __init__(self, input_size, output_size, h1=32, h2=64, h3=128, h4=256):
        super(LeaseModel, self).__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.out = nn.Linear(h4, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.2)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, p=0.2)
        x = self.out(x)
        return x