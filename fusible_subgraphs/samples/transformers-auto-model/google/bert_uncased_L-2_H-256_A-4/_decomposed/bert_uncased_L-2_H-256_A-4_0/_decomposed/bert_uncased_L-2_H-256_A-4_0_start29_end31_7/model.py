import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1):
        tmp_0 = in_1 + in_0
        tmp_1 = torch.nn.functional.layer_norm(tmp_0, (256,), w_1, w_0, 1e-12)
        tmp_0 = None
        return (tmp_1,)