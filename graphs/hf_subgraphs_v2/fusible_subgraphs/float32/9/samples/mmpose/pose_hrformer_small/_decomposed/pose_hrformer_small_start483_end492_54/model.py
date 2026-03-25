import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
        tmp_1 = tmp_0.flatten(2)
        tmp_0 = None
        tmp_2 = tmp_1.transpose(1, 2)
        tmp_1 = None
        tmp_3 = tmp_2.contiguous()
        tmp_2 = None
        tmp_4 = in_2 + tmp_3
        tmp_3 = None
        tmp_5 = tmp_4.permute(0, 2, 1)
        tmp_4 = None
        tmp_6 = tmp_5.view(1, 64, 32, 24)
        tmp_5 = None
        tmp_7 = in_1.view(1, 128, -1)
        tmp_8 = tmp_7.permute(0, 2, 1)
        tmp_7 = None
        return (tmp_6, tmp_8)