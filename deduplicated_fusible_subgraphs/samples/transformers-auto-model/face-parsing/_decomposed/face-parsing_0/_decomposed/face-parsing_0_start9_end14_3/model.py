import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1):
        tmp_0 = in_1 + in_0
        tmp_1 = torch.nn.functional.layer_norm(tmp_0, (64,), w_1, w_0, 1e-05)
        tmp_0 = None
        tmp_2 = tmp_1.reshape(1, 128, 128, -1)
        tmp_1 = None
        tmp_3 = tmp_2.permute(0, 3, 1, 2)
        tmp_2 = None
        tmp_4 = tmp_3.contiguous()
        tmp_3 = None
        return (tmp_4,)