import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0.reshape(4, 32, 512, 64)
        tmp_1 = in_1[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
        tmp_2 = tmp_1.expand(4, 4, 8, 512, 64)
        tmp_1 = None
        return (tmp_2, tmp_0)