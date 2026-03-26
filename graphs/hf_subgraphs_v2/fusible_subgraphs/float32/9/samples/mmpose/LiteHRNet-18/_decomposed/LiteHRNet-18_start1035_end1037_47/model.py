import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.interpolate(in_0, size=(32, 24), mode='nearest')
        tmp_1 = in_1 * tmp_0
        tmp_0 = None
        return (tmp_1,)