import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3 + in_4
        tmp_4 = torch.nn.functional.layer_norm(tmp_3, (512,), tmp_2, tmp_1, 1e-05)
        tmp_3 = tmp_2 = tmp_1 = None
        tmp_5 = tmp_0.view(-1, 1)
        tmp_0 = tmp_5 = None
        return (tmp_4,)