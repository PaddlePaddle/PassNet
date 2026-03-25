import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_3.contiguous()
        tmp_3 = tmp_2.view(-1, 4, 4, 64)
        tmp_2 = None
        tmp_4 = tmp_3.view(1, 16, 64)
        tmp_3 = None
        tmp_5 = torch.nn.functional.layer_norm(tmp_4, (64,), tmp_1, tmp_0, 1e-05)
        tmp_4 = tmp_1 = tmp_0 = None
        tmp_6 = in_2 + tmp_5
        tmp_5 = None
        return (tmp_6,)