import torch

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.zeros((1, 70, 70, 1))
        return (tmp_0,)