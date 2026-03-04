import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0 + in_3
        tmp_1 = tmp_0 + in_2
        tmp_0 = None
        tmp_2 = tmp_1 / 8.0
        tmp_1 = None
        tmp_3 = tmp_2 + in_1
        tmp_2 = None
        tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
        tmp_3 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.1, False, False)
        tmp_4 = None
        return (tmp_5,)