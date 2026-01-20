import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, w_0, w_1):
        tmp_0 = torch.nn.functional.linear(in_0, w_1, w_0)
        tmp_1 = in_2.view(1, 22, -1, 4)
        tmp_2 = tmp_1.transpose(1, 2)
        tmp_1 = None
        tmp_3 = tmp_0.view(1, 22, -1, 4)
        tmp_0 = None
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = in_1[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 22, None)]
        return (tmp_2, tmp_4, tmp_5)