import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1):
        tmp_0 = torch.nn.functional.dropout(in_0, 0.1, False, False)
        tmp_1 = torch.nn.functional.linear(tmp_0, w_1, w_0)
        tmp_0 = None
        return (tmp_1,)