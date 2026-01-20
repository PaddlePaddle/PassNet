import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1):
        tmp_0 = in_1.contiguous()
        tmp_1 = tmp_0.view(-1, 8, 8, 32)
        tmp_0 = None
        tmp_2 = torch.roll(tmp_1, shifts=(1, 1), dims=(1, 2))
        tmp_1 = None
        tmp_3 = tmp_2.view(1, 64, 32)
        tmp_2 = None
        tmp_4 = in_0 + tmp_3
        tmp_3 = None
        tmp_5 = torch.nn.functional.layer_norm(tmp_4, (32,), w_1, w_0, 1e-05)
        return (tmp_4, tmp_5)