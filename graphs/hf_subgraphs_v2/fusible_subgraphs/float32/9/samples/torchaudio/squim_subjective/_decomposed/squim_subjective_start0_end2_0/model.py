import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = tmp_0[slice(None, None, None), slice(None, 80000, None)]
        tmp_0 = None
        tmp_3 = tmp_1.unsqueeze(1)
        tmp_1 = None
        return (tmp_2, tmp_3)