import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.linear(in_3, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.view((1, 64, -1, 80))
        tmp_2 = None
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = in_5[Ellipsis, slice(None, 32, None)]
        tmp_6 = in_5[Ellipsis, slice(32, None, None)]
        tmp_7 = in_4[Ellipsis, slice(None, 32, None)]
        tmp_8 = in_4[Ellipsis, slice(32, None, None)]
        tmp_9 = in_2.unsqueeze(1)
        tmp_10 = in_6.unsqueeze(1)
        return (tmp_9, tmp_8, tmp_7, tmp_6, tmp_5, tmp_10, tmp_4)