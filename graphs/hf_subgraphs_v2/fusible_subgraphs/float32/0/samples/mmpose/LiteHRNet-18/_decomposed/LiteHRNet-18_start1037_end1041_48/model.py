import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = torch.nn.functional.interpolate(in_0, size=(16, 12), mode='nearest')
        tmp_1 = in_2 * tmp_0
        tmp_0 = None
        tmp_2 = torch.nn.functional.interpolate(in_1, size=(8, 6), mode='nearest')
        tmp_3 = in_3 * tmp_2
        tmp_2 = None
        return (tmp_1, tmp_3)