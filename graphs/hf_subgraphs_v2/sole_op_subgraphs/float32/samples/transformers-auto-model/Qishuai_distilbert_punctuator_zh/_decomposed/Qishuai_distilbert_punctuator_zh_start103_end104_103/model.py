import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.dropout(in_1, in_0, False, False)
        return (tmp_0,)