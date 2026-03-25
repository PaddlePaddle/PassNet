import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        in_5 += in_0
        tmp_0 = in_5
        tmp_0 += in_1
        tmp_1 = tmp_0
        tmp_0 = None
        tmp_2 = torch.nn.functional.relu(tmp_1, inplace=False)
        tmp_1 = None
        tmp_3 = torch.nn.functional.interpolate(in_2, (128, 128), None, 'bilinear', False)
        tmp_4 = torch.nn.functional.interpolate(in_3, (128, 128), None, 'bilinear', False)
        tmp_5 = torch.nn.functional.interpolate(in_4, (128, 128), None, 'bilinear', False)
        tmp_6 = torch.nn.functional.interpolate(tmp_2, (128, 128), None, 'bilinear', False)
        tmp_7 = torch.cat([tmp_3, tmp_4, tmp_5, tmp_6], dim=1)
        tmp_3 = tmp_4 = tmp_5 = tmp_6 = None
        return (tmp_7, tmp_2)