import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = torch.nn.functional.interpolate(in_2, (64, 64), None, 'bilinear', False)
        tmp_1 = torch.sigmoid(tmp_0)
        tmp_0 = None
        tmp_2 = in_1 * tmp_1
        tmp_1 = None
        tmp_3 = torch.sigmoid(in_3)
        tmp_4 = in_0 * tmp_3
        tmp_3 = None
        tmp_5 = torch.nn.functional.interpolate(tmp_4, (64, 64), None, 'bilinear', False)
        tmp_4 = None
        return (tmp_2, tmp_5)