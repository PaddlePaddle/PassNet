import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = torch.cat((in_2, in_3), 1)
        tmp_1 = torch.nn.functional.interpolate(in_0, size=(40, 40), mode='nearest')
        tmp_2 = torch.nn.functional.interpolate(in_1, size=(40, 40), mode='nearest')
        tmp_3 = torch.stack([tmp_1, tmp_2, tmp_0])
        tmp_1 = tmp_2 = tmp_0 = None
        return (tmp_3,)