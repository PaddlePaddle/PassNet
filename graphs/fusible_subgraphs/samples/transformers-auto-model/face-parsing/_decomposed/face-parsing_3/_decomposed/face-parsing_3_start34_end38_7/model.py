import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1):
        tmp_0 = torch.nn.functional.linear(in_0, w_1, w_0)
        tmp_1 = tmp_0.permute(0, 2, 1)
        tmp_0 = None
        tmp_2 = tmp_1.reshape(1, -1, 16, 16)
        tmp_1 = None
        tmp_3 = torch.nn.functional.interpolate(tmp_2, size=(128, 128), mode='bilinear', align_corners=False)
        tmp_2 = None
        return (tmp_3,)