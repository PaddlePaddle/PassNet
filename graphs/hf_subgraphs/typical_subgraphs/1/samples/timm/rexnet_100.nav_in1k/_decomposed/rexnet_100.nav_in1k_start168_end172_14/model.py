import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0[slice(None, None, None), slice(0, 117, None)]
        tmp_1 = tmp_0 + in_1
        tmp_0 = None
        tmp_2 = in_0[slice(None, None, None), slice(117, None, None)]
        tmp_3 = torch.cat([tmp_1, tmp_2], dim=1)
        tmp_1 = tmp_2 = None
        return (tmp_3,)