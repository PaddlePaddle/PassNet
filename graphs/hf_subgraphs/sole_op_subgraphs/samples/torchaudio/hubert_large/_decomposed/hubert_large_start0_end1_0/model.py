import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0
        tmp_1 = torch.nn.functional.layer_norm(tmp_0, (1, 80000))
        tmp_0 = None
        return (tmp_1,)