import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_3.contiguous()
        tmp_3 = tmp_2.view(-1, 16, 16, 16)
        tmp_2 = None
        tmp_4 = tmp_3.view(1, 256, 16)
        tmp_3 = None
        tmp_5 = in_2 + tmp_4
        tmp_4 = None
        tmp_6 = torch.nn.functional.layer_norm(tmp_5, (16,), tmp_1, tmp_0, 1e-05)
        tmp_1 = tmp_0 = None
        return (tmp_5, tmp_6)