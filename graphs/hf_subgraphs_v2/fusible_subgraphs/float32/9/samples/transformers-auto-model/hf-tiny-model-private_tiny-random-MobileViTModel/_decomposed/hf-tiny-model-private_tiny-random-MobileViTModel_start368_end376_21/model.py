import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_3 + in_2
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (240,), tmp_1, tmp_0, 1e-05)
        tmp_2 = tmp_1 = tmp_0 = None
        tmp_4 = tmp_3.contiguous()
        tmp_3 = None
        tmp_5 = tmp_4.view(1, 4, 1, -1)
        tmp_4 = None
        tmp_6 = tmp_5.transpose(1, 3)
        tmp_5 = None
        tmp_7 = tmp_6.reshape(240, 1, 2, 2)
        tmp_6 = None
        tmp_8 = tmp_7.transpose(1, 2)
        tmp_7 = None
        tmp_9 = tmp_8.reshape(1, 240, 2, 2)
        tmp_8 = None
        return (tmp_9,)