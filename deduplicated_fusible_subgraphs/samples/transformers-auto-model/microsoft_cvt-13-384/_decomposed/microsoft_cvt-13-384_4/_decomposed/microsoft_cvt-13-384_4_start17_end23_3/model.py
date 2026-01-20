import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1):
        tmp_0 = torch.nn.functional.layer_norm(in_0, (384,), w_1, w_0, 1e-05)
        tmp_1 = torch.functional.split(tmp_0, [1, 576], 1)
        tmp_0 = None
        tmp_2 = tmp_1[0]
        tmp_3 = tmp_1[1]
        tmp_1 = None
        tmp_4 = tmp_3.permute(0, 2, 1)
        tmp_3 = None
        tmp_5 = tmp_4.view(1, 384, 24, 24)
        tmp_4 = None
        return (tmp_2, tmp_5)