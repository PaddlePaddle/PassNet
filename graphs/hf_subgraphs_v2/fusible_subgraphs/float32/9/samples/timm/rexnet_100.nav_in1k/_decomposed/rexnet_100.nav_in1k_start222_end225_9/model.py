import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0 + in_1
        tmp_1 = in_2[slice(None, None, None), slice(151, None, None)]
        tmp_2 = torch.cat([tmp_0, tmp_1], dim=1)
        tmp_0 = tmp_1 = None
        return (tmp_2,)