import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0
        tmp_1 = tmp_0[slice(None, None, None), 0]
        tmp_0 = None
        tmp_2 = torch.unsqueeze(tmp_1, 1)
        tmp_1 = None
        return (tmp_2,)