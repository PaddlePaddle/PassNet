import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = torch.matmul(in_1, in_2)
        tmp_2 = tmp_0[slice(None, 45, None), slice(None, 45, None), slice(None, None, None)]
        tmp_0 = None
        tmp_3 = in_1.permute(2, 0, 1, 3)
        tmp_4 = tmp_3.contiguous()
        tmp_3 = None
        tmp_5 = tmp_4.view(45, 4, 8)
        tmp_4 = None
        tmp_6 = tmp_2.permute(0, 2, 1)
        tmp_2 = None
        return (tmp_1, tmp_6, tmp_5)