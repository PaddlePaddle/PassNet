import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.cat([in_0, in_1], dim=1)
        tmp_1 = tmp_0[slice(None, None, None), slice(None, 120, None), slice(None, None, None), slice(None, None, None)]
        tmp_0 = None
        tmp_2 = tmp_1.mean((2, 3), keepdim=True)
        return (tmp_1, tmp_2)