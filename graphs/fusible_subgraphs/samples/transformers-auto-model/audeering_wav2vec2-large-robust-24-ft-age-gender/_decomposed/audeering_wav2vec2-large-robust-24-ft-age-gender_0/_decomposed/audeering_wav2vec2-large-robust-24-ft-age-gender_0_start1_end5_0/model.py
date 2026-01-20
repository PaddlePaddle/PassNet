import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3):
        tmp_0 = torch.conv1d(in_0, w_1, w_0, (5,), (0,), (1,), 1)
        tmp_1 = tmp_0.transpose(-2, -1)
        tmp_0 = None
        tmp_2 = torch.nn.functional.layer_norm(tmp_1, (512,), w_3, w_2, 1e-05)
        tmp_1 = None
        tmp_3 = tmp_2.transpose(-2, -1)
        tmp_2 = None
        return (tmp_3,)