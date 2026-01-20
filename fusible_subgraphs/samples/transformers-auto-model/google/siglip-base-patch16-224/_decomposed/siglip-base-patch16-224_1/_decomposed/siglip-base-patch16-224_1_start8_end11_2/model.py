import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1):
        tmp_0 = in_0 + in_1
        tmp_1 = torch.nn.functional.layer_norm(tmp_0, (768,), w_1, w_0, 1e-06)
        tmp_0 = None
        tmp_2 = tmp_1[slice(None, None, None), -1, slice(None, None, None)]
        return (tmp_1, tmp_2)