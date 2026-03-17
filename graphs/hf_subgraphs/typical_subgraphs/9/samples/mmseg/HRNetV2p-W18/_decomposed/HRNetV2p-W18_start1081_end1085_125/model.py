import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.interpolate(in_0, [128, 128], None, 'bilinear', False)
        tmp_1 = torch.nn.functional.interpolate(tmp_0, (128, 128), None, 'bilinear', False)
        tmp_0 = None
        tmp_2 = in_1 + tmp_1
        tmp_1 = None
        tmp_3 = torch.nn.functional.relu(tmp_2, inplace=False)
        tmp_2 = None
        return (tmp_3,)