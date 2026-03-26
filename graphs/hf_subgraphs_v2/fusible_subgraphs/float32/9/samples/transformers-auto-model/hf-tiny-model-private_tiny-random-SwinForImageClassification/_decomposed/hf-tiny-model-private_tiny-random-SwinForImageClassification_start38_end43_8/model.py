import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
        tmp_2 = None
        tmp_4 = tmp_3.view(-1, 2, 2, 16)
        tmp_3 = None
        tmp_5 = tmp_4.view(-1, 8, 8, 2, 2, 16)
        tmp_4 = None
        tmp_6 = tmp_5.permute(0, 1, 3, 2, 4, 5)
        tmp_5 = None
        return (tmp_6,)