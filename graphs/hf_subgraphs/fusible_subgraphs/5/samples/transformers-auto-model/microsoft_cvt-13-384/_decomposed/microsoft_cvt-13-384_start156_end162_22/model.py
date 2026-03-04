import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.layer_norm(in_2, (384,), tmp_1, tmp_0, 1e-05)
        tmp_1 = tmp_0 = None
        tmp_3 = torch.functional.split(tmp_2, [1, 576], 1)
        tmp_2 = None
        tmp_4 = tmp_3[0]
        tmp_5 = tmp_3[1]
        tmp_3 = None
        tmp_6 = tmp_5.permute(0, 2, 1)
        tmp_5 = None
        tmp_7 = tmp_6.view(1, 384, 24, 24)
        tmp_6 = None
        return (tmp_4, tmp_7)