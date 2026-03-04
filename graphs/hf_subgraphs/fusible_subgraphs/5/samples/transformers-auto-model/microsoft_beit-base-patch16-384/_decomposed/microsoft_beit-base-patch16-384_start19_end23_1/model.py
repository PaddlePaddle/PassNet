import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = torch.nn.functional.interpolate(in_1, size=(47, 47), mode='bilinear')
        tmp_2 = tmp_1.permute(0, 2, 3, 1)
        tmp_1 = None
        tmp_3 = tmp_2.reshape(2209, -1)
        tmp_2 = None
        tmp_4 = tmp_0[slice(2209, None, None)]
        tmp_0 = None
        return (tmp_4, tmp_3)