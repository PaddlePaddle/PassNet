import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        in_2 += in_1
        tmp_0 = in_2
        tmp_1 = torch.nn.functional.relu(tmp_0, inplace=True)
        tmp_0 = None
        tmp_2 = torch.nn.functional.interpolate(tmp_1, None, 2.0, 'nearest', None, recompute_scale_factor=None)
        tmp_1 = None
        tmp_3 = in_0 + tmp_2
        tmp_2 = None
        return (tmp_3,)