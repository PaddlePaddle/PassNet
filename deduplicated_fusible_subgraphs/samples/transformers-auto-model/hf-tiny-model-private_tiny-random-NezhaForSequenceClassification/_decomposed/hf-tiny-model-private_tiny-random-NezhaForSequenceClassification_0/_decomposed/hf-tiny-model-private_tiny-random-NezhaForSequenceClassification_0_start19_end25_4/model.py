import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0):
        tmp_0 = torch.matmul(in_0, in_1)
        tmp_1 = w_0[slice(None, 45, None), slice(None, 45, None), slice(None, None, None)]
        tmp_2 = in_0.permute(2, 0, 1, 3)
        tmp_3 = tmp_2.contiguous()
        tmp_2 = None
        tmp_4 = tmp_3.view(45, 4, 8)
        tmp_3 = None
        tmp_5 = tmp_1.permute(0, 2, 1)
        tmp_1 = None
        return (tmp_0, tmp_4, tmp_5)