import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.layer_norm(in_0, (768,), None, None, 1e-06)
        return (tmp_0,)