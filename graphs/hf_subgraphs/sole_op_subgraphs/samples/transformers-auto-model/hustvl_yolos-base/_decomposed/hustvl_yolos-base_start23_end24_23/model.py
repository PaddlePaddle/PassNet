import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.view(11, 768, 50, 84)
        return (tmp_0,)