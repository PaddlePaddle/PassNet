import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0):
        tmp_0 = torch.matmul(in_1, in_0)
        tmp_1 = w_0[slice(None, 45, None), slice(None, 45, None), slice(None, None, None)]
        tmp_2 = in_1.permute(2, 0, 1, 3)
        return (tmp_0, tmp_1, tmp_2)