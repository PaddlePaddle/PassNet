import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1):
        tmp_0 = torch.nn.functional.layer_norm(in_0, (512,), w_1, w_0, 1e-05)
        tmp_1 = tmp_0.transpose(-2, -1)
        tmp_0 = None
        tmp_2 = torch.nn.functional.gelu(tmp_1)
        tmp_1 = None
        return (tmp_2,)