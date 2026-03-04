import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_3 + in_2
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (384,), tmp_1, tmp_0, 1e-05)
        tmp_1 = tmp_0 = None
        tmp_4 = torch.functional.split(tmp_3, [1, 196], 1)
        tmp_3 = None
        tmp_5 = tmp_4[0]
        tmp_6 = tmp_4[1]
        tmp_4 = None
        tmp_7 = tmp_6.permute(0, 2, 1)
        tmp_6 = None
        tmp_8 = tmp_7.view(1, 384, 14, 14)
        tmp_7 = None
        return (tmp_5, tmp_2, tmp_8)