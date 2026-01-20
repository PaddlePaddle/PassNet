import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1):
        tmp_0 = torch.nn.functional.linear(in_0, w_1, w_0)
        tmp_1 = torch.nn.functional.dropout(tmp_0, 0.0, False, False)
        tmp_0 = None
        tmp_2 = tmp_1.view(-1, 2, 2, 64)
        tmp_1 = None
        tmp_3 = tmp_2.view(-1, 2, 2, 2, 2, 64)
        tmp_2 = None
        tmp_4 = tmp_3.permute(0, 1, 3, 2, 4, 5)
        tmp_3 = None
        return (tmp_4,)