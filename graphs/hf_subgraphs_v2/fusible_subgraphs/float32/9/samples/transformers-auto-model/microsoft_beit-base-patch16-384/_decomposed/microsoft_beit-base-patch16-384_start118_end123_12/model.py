import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0[in_1]
        tmp_1 = tmp_0.view(577, 577, -1)
        tmp_0 = None
        tmp_2 = tmp_1.permute(2, 0, 1)
        tmp_1 = None
        tmp_3 = tmp_2.contiguous()
        tmp_2 = None
        tmp_4 = tmp_3.unsqueeze(0)
        tmp_3 = None
        return (tmp_4,)