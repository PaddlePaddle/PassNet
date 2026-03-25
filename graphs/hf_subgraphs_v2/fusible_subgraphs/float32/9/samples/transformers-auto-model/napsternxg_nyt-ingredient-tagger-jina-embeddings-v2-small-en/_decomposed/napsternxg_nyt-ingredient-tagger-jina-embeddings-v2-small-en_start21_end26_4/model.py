import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_1 / 8.0
        tmp_1 = tmp_0 + in_2
        tmp_0 = None
        tmp_2 = tmp_1 + in_0
        tmp_1 = None
        tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)
        tmp_2 = None
        tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
        tmp_3 = None
        return (tmp_4,)