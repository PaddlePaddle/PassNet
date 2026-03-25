import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.interpolate(in_0, size=(2, 2), mode='bilinear')
        tmp_1 = tmp_0.permute(0, 2, 3, 1)
        tmp_0 = None
        tmp_2 = tmp_1.reshape(1, 4, -1)
        tmp_1 = None
        return (tmp_2,)