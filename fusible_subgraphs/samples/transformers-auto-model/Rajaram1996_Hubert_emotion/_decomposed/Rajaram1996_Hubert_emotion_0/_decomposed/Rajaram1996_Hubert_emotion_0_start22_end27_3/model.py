import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, w_0):
        tmp_0 = torch.conv1d(in_1, in_2, w_0, (1,), (64,), (1,), 16)
        tmp_1 = tmp_0[slice(None, None, None), slice(None, None, None), slice(None, -1, None)]
        tmp_0 = None
        tmp_2 = torch.nn.functional.gelu(tmp_1)
        tmp_1 = None
        tmp_3 = tmp_2.transpose(1, 2)
        tmp_2 = None
        tmp_4 = in_0 + tmp_3
        tmp_3 = None
        return (tmp_4,)