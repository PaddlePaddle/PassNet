import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_3.contiguous()
        tmp_3 = tmp_2.view(-1, 14, 14, 384)
        tmp_2 = None
        tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
        tmp_3 = None
        tmp_5 = tmp_4.view(1, 196, 384)
        tmp_4 = None
        tmp_6 = in_2 + tmp_5
        tmp_5 = None
        tmp_7 = torch.nn.functional.layer_norm(tmp_6, (384,), tmp_1, tmp_0, 1e-05)
        tmp_1 = tmp_0 = None
        return (tmp_6, tmp_7)