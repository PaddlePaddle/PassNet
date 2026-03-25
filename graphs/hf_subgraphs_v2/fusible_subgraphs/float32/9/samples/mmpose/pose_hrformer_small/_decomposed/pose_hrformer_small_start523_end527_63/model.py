import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0.reshape(1, 21, 14, 128)
        tmp_1 = tmp_0[slice(None, None, None), slice(2, 18, None), slice(1, 13, None)]
        tmp_0 = None
        tmp_2 = tmp_1.reshape(1, 192, 128)
        tmp_1 = None
        tmp_3 = in_1 + tmp_2
        tmp_2 = None
        return (tmp_3,)