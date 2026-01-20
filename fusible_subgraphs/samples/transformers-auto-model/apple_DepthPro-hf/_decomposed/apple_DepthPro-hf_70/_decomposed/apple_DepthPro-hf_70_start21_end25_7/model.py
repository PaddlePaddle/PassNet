import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1):
        tmp_0 = torch.nn.functional.linear(in_0, w_1, w_0)
        tmp_1 = tmp_0[slice(None, None, None), slice(-576, None, None), slice(None, None, None)]
        tmp_0 = None
        tmp_2 = tmp_1.reshape(1, 24, 24, 128)
        tmp_1 = None
        tmp_3 = tmp_2.permute(0, 3, 1, 2)
        tmp_2 = None
        return (tmp_3,)