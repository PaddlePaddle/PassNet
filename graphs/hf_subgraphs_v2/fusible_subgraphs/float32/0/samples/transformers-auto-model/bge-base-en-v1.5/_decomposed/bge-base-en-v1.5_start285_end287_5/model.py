import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.cat([in_0], 1)
        tmp_1 = torch.nn.functional.normalize(tmp_0, p=2, dim=1)
        tmp_0 = None
        return (tmp_1,)