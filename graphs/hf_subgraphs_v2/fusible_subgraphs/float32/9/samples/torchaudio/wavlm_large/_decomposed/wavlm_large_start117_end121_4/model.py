import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2 + in_3
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (1024,), tmp_1, tmp_0, 1e-05)
        tmp_1 = tmp_0 = None
        tmp_4 = tmp_3.view(1, 199, 16, -1)
        tmp_5 = tmp_4.permute(0, 2, 1, 3)
        tmp_4 = None
        return (tmp_5, tmp_2, tmp_3)