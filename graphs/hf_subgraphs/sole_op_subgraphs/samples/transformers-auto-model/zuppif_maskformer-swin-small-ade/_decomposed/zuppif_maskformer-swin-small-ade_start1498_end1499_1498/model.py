import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.view(-1, 3, 3, 7, 7, 768)
        return (tmp_0,)