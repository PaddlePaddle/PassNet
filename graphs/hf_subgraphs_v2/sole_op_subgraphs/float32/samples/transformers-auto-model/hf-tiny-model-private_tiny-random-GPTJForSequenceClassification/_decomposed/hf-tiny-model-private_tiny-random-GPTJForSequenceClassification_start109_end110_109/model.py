import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.gather(in_0, 1, in_1)
        return (tmp_0,)