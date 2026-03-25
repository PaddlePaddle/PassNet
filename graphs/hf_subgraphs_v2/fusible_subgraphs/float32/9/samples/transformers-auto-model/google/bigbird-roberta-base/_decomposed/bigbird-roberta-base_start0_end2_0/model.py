import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
        tmp_2 = None
        tmp_4 = torch.nn.functional.linear(tmp_3, tmp_1, tmp_0)
        tmp_3 = tmp_1 = tmp_0 = None
        return (tmp_4,)