import torch
import torch.multiprocessing as mp
from torch import nn
import os

class LatentClassifier(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_nodes):
        super(LatentClassifier, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_nodes = hidden_nodes

        self.linear1 = nn.Linear(input_shape, hidden_nodes)
        self.linear2 = nn.Linear(hidden_nodes, int(hidden_nodes / 2))
        self.linear3 = nn.Linear(int(hidden_nodes / 2), int(hidden_nodes / 8))
        # self.linear4 = nn.Linear(int(hidden_nodes/4), int(hidden_nodes/8))
        self.linear5 = nn.Linear(int(hidden_nodes / 8), int(hidden_nodes / 32))
        # self.linear6 = nn.Linear(int(hidden_nodes/16), int(hidden_nodes/32))
        self.linear7 = nn.Linear(int(hidden_nodes / 32), output_shape)
        self.bn3 = nn.BatchNorm1d(num_features=int(hidden_nodes / 8))
        self.bn5 = nn.BatchNorm1d(num_features=int(hidden_nodes / 32))
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, input):
        x = self.lrelu(self.linear1(input))
        x = self.lrelu(self.linear2(x))
        x = self.lrelu(self.bn3(self.linear3(x)))
        # x = self.lrelu(self.linear4(x))
        x = self.lrelu(self.bn5(self.linear5(x)))
        # x = self.lrelu(self.linear6(x))
        # x = self.sigmoid(self.linear7(x))
        x = self.linear7(x)
        return x.squeeze()

    def weights_init(self, m):
        torch.init.xavier_uniform_(m.weight.data)

def train(model, optimizer, loss_fn):
    # Construct data_loader, optimizer, etc.
    for _ in range(1000):
        data = torch.randn((32, 100)).to('cuda')
        labels = torch.randn((32, 2)).to('cuda')
        optimizer.zero_grad()
        loss = loss_fn(model(data), labels)
        loss.backward()
        optimizer.step()  # This will update the shared parameters
        if _ % 10 == 0:
            print(f'{os.getpid()}-loss={loss.item()}')

if __name__ == '__main__':
    mp.set_start_method('spawn')
    num_processes = 2
    model = LatentClassifier(100, 2, 500)
    model = model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()
    # NOTE: this is required for the ``fork`` method to work
    # model.share_memory()
    print(model)

    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(model, optimizer, loss_fn))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()