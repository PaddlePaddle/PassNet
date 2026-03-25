import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.sigmoid(in_1)
        tmp_1 = in_1 * tmp_0
        tmp_0 = None
        tmp_2 = torch.nn.functional.interpolate(tmp_1, None, 2.0, 'nearest', None, recompute_scale_factor=None)
        tmp_3 = torch.cat([tmp_2, in_0], 1)
        tmp_2 = None
        return (tmp_3, tmp_1)