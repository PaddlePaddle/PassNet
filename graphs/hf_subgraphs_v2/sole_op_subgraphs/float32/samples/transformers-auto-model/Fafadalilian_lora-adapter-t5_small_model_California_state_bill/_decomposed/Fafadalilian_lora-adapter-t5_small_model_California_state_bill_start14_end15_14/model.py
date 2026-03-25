import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = torch.nn.functional.dropout(tmp_0, in_1, False, False)
        tmp_0 = None
        return (tmp_1,)