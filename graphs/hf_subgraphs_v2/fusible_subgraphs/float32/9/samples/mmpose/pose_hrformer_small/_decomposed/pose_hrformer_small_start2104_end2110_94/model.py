import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
        tmp_2 = None
        tmp_4 = tmp_3.reshape(1, 2, 1, 7, 7, 256)
        tmp_3 = None
        tmp_5 = tmp_4.permute(0, 1, 3, 2, 4, 5)
        tmp_4 = None
        tmp_6 = tmp_5.reshape(1, 14, 7, 256)
        tmp_5 = None
        tmp_7 = tmp_6[slice(None, None, None), slice(3, 11, None), slice(0, 6, None)]
        tmp_6 = None
        return (tmp_7,)