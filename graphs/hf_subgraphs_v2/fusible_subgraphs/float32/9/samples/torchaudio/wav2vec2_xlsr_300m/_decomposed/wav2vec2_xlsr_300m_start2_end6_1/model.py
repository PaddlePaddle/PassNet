import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.conv1d(in_4, tmp_1, tmp_0, (5,), (0,), (1,), 1)
        tmp_1 = tmp_0 = None
        tmp_5 = tmp_4.transpose(-2, -1)
        tmp_4 = None
        tmp_6 = torch.nn.functional.layer_norm(tmp_5, (512,), tmp_3, tmp_2, 1e-05)
        tmp_5 = tmp_3 = tmp_2 = None
        tmp_7 = tmp_6.transpose(-2, -1)
        tmp_6 = None
        return (tmp_7,)