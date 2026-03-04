import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2 + in_3
        tmp_3 = tmp_2[slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_4 = tmp_3.reshape(1, 12, 12, -1)
        tmp_3 = None
        tmp_5 = tmp_4.permute(0, 3, 1, 2)
        tmp_4 = None
        tmp_6 = tmp_5.contiguous()
        tmp_5 = tmp_6 = None
        tmp_7 = torch.nn.functional.layer_norm(tmp_2, (512,), tmp_1, tmp_0, 1e-06)
        tmp_2 = tmp_1 = tmp_0 = None
        tmp_8 = tmp_7[slice(None, None, None), 0]
        tmp_7 = None
        return (tmp_8,)