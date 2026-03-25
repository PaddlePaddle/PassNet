import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
        tmp_2 = None
        tmp_4 = tmp_3.reshape(1, 3, 2, 7, 7, 128)
        tmp_3 = None
        tmp_5 = tmp_4.permute(0, 1, 3, 2, 4, 5)
        tmp_4 = None
        return (tmp_5,)