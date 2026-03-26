import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = torch.nn.functional.relu(in_3, inplace=False)
        tmp_1 = torch.nn.functional.interpolate(in_0, (32, 32), None, 'bilinear', False)
        tmp_2 = torch.nn.functional.interpolate(tmp_0, (32, 32), None, 'bilinear', False)
        tmp_0 = None
        tmp_3 = torch.nn.functional.interpolate(in_2, (32, 32), None, 'bilinear', False)
        tmp_4 = torch.cat([in_1, tmp_3, tmp_2, tmp_1], dim=1)
        tmp_3 = tmp_2 = tmp_1 = None
        return (tmp_4,)