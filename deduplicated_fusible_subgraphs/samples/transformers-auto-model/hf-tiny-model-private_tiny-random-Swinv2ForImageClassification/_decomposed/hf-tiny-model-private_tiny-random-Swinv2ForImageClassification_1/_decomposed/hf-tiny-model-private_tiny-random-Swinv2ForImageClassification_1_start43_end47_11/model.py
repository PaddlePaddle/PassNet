import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2, w_3):
        tmp_0 = torch.nn.functional.layer_norm(in_1, (64,), w_1, w_0, 1e-05)
        tmp_1 = in_0 + tmp_0
        tmp_0 = None
        tmp_2 = torch.nn.functional.layer_norm(tmp_1, (64,), w_3, w_2, 1e-05)
        tmp_1 = None
        tmp_3 = tmp_2.transpose(1, 2)
        tmp_2 = None
        return (tmp_3,)