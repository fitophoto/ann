from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
           nn.Linear(50 * 2 + 1, 200),
           nn.ReLU(),
           nn.Linear(200, 150),
           nn.ReLU(),
           nn.Linear(150, 3),
        )
    def forward(self, x):
        return self.linear(x)