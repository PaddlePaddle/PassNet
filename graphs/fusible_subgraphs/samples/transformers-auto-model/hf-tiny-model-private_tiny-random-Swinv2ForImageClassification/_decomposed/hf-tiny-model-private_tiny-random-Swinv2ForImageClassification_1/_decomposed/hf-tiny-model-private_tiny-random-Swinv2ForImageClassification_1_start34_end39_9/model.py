import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1):
        tmp_0 = in_1.contiguous()
        tmp_1 = tmp_0.view(-1, 4, 4, 64)
        tmp_0 = None
        tmp_2 = tmp_1.view(1, 16, 64)
        tmp_1 = None
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (64,), w_1, w_0, 1e-05)
        tmp_2 = None
        tmp_4 = in_0 + tmp_3
        tmp_3 = None
        return (tmp_4,)