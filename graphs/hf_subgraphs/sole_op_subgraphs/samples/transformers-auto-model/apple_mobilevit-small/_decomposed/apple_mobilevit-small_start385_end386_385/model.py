import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.mean(in_0, dim=[-2, -1], keepdim=False)
        return (tmp_0,)