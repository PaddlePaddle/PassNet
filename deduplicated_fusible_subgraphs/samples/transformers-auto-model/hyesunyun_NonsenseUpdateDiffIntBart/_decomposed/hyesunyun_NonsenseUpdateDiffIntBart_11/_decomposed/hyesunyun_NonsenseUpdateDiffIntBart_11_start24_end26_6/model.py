import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1):
        tmp_0 = in_0 + in_1
        tmp_1 = torch.nn.functional.layer_norm(tmp_0, (1024,), w_1, w_0, 1e-05)
        tmp_0 = None
        return (tmp_1,)