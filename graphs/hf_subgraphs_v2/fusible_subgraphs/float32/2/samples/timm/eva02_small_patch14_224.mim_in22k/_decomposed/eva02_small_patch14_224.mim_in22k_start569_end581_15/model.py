import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.linear(in_3, weight=tmp_0, bias=in_2)
        tmp_0 = None
        tmp_3 = tmp_2.reshape(1, 257, 3, 6, -1)
        tmp_2 = None
        tmp_4 = tmp_3.permute(2, 0, 3, 1, 4)
        tmp_3 = None
        tmp_5 = tmp_4.unbind(0)
        tmp_4 = None
        tmp_6 = tmp_5[0]
        tmp_7 = tmp_5[1]
        tmp_8 = tmp_5[2]
        tmp_5 = None
        tmp_9 = tmp_6[slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None)]
        tmp_10 = tmp_6[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_6 = None
        tmp_11 = tmp_1.tensor_split(2, -1)
        tmp_1 = None
        tmp_12 = tmp_11[0]
        tmp_13 = tmp_11[1]
        tmp_11 = None
        return (tmp_13, tmp_9, tmp_10, tmp_7, tmp_12, tmp_8)