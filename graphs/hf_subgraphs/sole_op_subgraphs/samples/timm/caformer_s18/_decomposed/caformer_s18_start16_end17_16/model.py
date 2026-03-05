import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, in_0):
        tmp_0 = w_0
        tmp_1 = torch.nn.functional.layer_norm(in_0, (64,), tmp_0, None, 1e-06)
        tmp_0 = None
        return (tmp_1,)