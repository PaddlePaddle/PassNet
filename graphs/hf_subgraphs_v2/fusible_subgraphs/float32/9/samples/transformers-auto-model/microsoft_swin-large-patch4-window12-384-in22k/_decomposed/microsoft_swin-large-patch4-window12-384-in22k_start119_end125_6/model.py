import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_3.contiguous()
        tmp_3 = tmp_2.view(-1, 96, 96, 192)
        tmp_2 = None
        tmp_4 = torch.roll(tmp_3, shifts=(6, 6), dims=(1, 2))
        tmp_3 = None
        tmp_5 = tmp_4.view(1, 9216, 192)
        tmp_4 = None
        tmp_6 = in_2 + tmp_5
        tmp_5 = None
        tmp_7 = torch.nn.functional.layer_norm(tmp_6, (192,), tmp_1, tmp_0, 1e-05)
        tmp_1 = tmp_0 = None
        return (tmp_6, tmp_7)