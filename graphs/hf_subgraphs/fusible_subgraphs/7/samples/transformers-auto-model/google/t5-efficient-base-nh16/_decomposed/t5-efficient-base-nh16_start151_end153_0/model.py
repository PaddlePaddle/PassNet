import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = torch.nn.functional.linear(in_1, tmp_0, None)
        tmp_0 = None
        tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
        tmp_1 = None
        return (tmp_2,)