import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.transpose(in_0, 1, 2)
        return (tmp_0,)