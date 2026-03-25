import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = torch.nn.functional.interpolate(in_0, size=(160, 160), mode='nearest')
        tmp_1 = torch.nn.functional.interpolate(in_1, size=(160, 160), mode='nearest')
        tmp_2 = torch.nn.functional.interpolate(in_2, size=(160, 160), mode='nearest')
        tmp_3 = torch.nn.functional.interpolate(in_3, size=(160, 160), mode='nearest')
        tmp_4 = torch.stack([tmp_0, tmp_1, tmp_2, tmp_3, in_4])
        tmp_0 = tmp_1 = tmp_2 = tmp_3 = None
        return (tmp_4,)