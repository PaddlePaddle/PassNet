import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.cat([in_2, in_3, in_4, in_5], -1)
        tmp_3 = tmp_2.view(1, -1, 384)
        tmp_2 = None
        tmp_4 = torch.nn.functional.layer_norm(tmp_3, (384,), tmp_1, tmp_0, 1e-05)
        tmp_3 = tmp_1 = tmp_0 = None
        return (tmp_4,)