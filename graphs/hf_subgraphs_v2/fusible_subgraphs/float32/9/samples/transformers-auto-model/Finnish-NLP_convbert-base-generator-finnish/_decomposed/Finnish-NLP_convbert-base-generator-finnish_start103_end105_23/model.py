import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.cat([in_0, in_1], 2)
        tmp_1 = tmp_0.view(1, 19, 256)
        tmp_0 = None
        return (tmp_1,)