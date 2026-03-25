import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.permute(0, 2, 1)
        tmp_2 = None
        tmp_4 = tmp_3.reshape(4, -1, 64, 64)
        tmp_3 = None
        tmp_5 = torch.nn.functional.interpolate(tmp_4, size=(128, 128), mode='bilinear', align_corners=False)
        tmp_4 = None
        tmp_6 = in_3.flatten(2)
        tmp_7 = tmp_6.transpose(1, 2)
        tmp_6 = None
        return (tmp_5, tmp_7)