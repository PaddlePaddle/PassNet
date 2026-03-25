import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = torch.conv1d(in_2, in_3, tmp_0, (1,), (64,), (1,), 16)
        tmp_0 = None
        tmp_2 = tmp_1[slice(None, None, None), slice(None, None, None), slice(None, -1, None)]
        tmp_1 = None
        tmp_3 = torch.nn.functional.gelu(tmp_2)
        tmp_2 = None
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = in_1 + tmp_4
        tmp_4 = None
        return (tmp_5,)