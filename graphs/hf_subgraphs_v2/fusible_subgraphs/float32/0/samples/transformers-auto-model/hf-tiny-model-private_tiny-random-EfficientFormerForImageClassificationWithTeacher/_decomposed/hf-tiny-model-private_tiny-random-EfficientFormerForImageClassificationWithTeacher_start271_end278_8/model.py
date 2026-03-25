import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.linear(in_5, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_5 = tmp_2.unsqueeze(0)
        tmp_2 = None
        tmp_6 = tmp_5.unsqueeze(0)
        tmp_5 = None
        tmp_7 = tmp_6 * tmp_4
        tmp_6 = tmp_4 = None
        tmp_8 = in_4 + tmp_7
        tmp_7 = None
        tmp_9 = tmp_3.unsqueeze(0)
        tmp_3 = None
        tmp_10 = tmp_9.unsqueeze(0)
        tmp_9 = None
        return (tmp_8, tmp_10)