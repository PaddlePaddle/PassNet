import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.hardswish(in_0, True)
        tmp_1 = torch.nn.functional.dropout(tmp_0, 0.2, False, True)
        tmp_0 = None
        return (tmp_1,)