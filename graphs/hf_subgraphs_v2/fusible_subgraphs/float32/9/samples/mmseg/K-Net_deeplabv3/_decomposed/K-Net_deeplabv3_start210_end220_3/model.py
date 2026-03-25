import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.linear(in_5, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_5 = tmp_4[slice(None, None, None), slice(None, 256, None)]
        tmp_6 = tmp_5.view(-1, 256)
        tmp_5 = None
        tmp_7 = tmp_4[slice(None, None, None), slice(-256, None, None)]
        tmp_4 = None
        tmp_8 = tmp_7.view(-1, 256)
        tmp_7 = None
        tmp_9 = in_4.reshape(300, -1, 256)
        tmp_10 = torch.nn.functional.linear(tmp_9, tmp_3, tmp_2)
        tmp_9 = tmp_3 = tmp_2 = None
        tmp_11 = tmp_10[Ellipsis, slice(None, 256, None)]
        tmp_12 = tmp_10[Ellipsis, slice(-256, None, None)]
        tmp_10 = None
        tmp_13 = tmp_6.unsqueeze(-2)
        tmp_6 = None
        return (tmp_11, tmp_12, tmp_8, tmp_13)