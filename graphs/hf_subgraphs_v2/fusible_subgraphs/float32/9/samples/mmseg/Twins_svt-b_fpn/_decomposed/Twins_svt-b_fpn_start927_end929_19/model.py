import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.dropout(in_0, 0.0, False, False)
        tmp_1 = 0.0 + tmp_0
        tmp_0 = None
        return (tmp_1,)