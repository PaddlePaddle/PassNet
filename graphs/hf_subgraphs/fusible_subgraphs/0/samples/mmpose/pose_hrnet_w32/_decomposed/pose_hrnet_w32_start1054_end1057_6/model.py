import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.interpolate(in_0, None, 8.0, 'nearest', None, recompute_scale_factor=None)
        in_1 += tmp_0
        tmp_1 = in_1
        tmp_0 = None
        tmp_2 = torch.nn.functional.relu(tmp_1, inplace=True)
        tmp_1 = None
        return (tmp_2,)