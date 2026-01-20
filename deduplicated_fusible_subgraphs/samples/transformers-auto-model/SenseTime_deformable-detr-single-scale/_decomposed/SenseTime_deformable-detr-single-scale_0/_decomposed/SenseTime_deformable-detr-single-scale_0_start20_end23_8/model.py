import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.stack([in_1, in_2], -1)
        tmp_1 = in_0[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), None, slice(None, None, None)]
        tmp_2 = tmp_0[None, None, None, slice(None, None, None), None, slice(None, None, None)]
        tmp_0 = None
        return (tmp_1, tmp_2)