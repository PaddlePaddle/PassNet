import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.view(-1, 16, 16, 12, 12, 128)
        return (tmp_0,)