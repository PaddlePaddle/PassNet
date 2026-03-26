import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_3 + in_2
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (256,), tmp_1, tmp_0, 1e-05)
        tmp_2 = tmp_1 = tmp_0 = None
        tmp_4 = tmp_3.reshape(32, 16, 16, -1)
        tmp_3 = None
        tmp_5 = tmp_4.permute(0, 3, 1, 2)
        tmp_4 = None
        tmp_6 = tmp_5.contiguous()
        tmp_5 = None
        return (tmp_6,)