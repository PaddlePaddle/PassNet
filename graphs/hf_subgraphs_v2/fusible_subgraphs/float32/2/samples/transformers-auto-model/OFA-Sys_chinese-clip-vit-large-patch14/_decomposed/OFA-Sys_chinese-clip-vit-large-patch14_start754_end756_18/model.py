import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = torch.nn.functional.layer_norm(in_3, (1024,), tmp_2, tmp_1, 1e-05)
        tmp_2 = tmp_1 = None
        tmp_4 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        return (tmp_4, tmp_3)