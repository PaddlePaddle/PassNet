import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.gelu(in_2, approximate='none')
        tmp_3 = tmp_2.flatten(2)
        tmp_2 = None
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = tmp_4.contiguous()
        tmp_4 = None
        tmp_6 = in_3 + tmp_5
        tmp_5 = None
        tmp_7 = tmp_6.permute(0, 2, 1)
        tmp_6 = None
        tmp_8 = tmp_7.view(1, 256, 8, 6)
        tmp_7 = None
        tmp_9 = tmp_8.view(1, 256, -1)
        tmp_8 = None
        tmp_10 = tmp_9.permute(0, 2, 1)
        tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (256,), tmp_1, tmp_0, 1e-06)
        tmp_1 = tmp_0 = None
        tmp_12 = tmp_11.view(1, 8, 6, 256)
        tmp_11 = None
        return (tmp_10, tmp_12)