import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_1 = tmp_0 + in_2
        tmp_0 = None
        tmp_2 = torch.nn.functional.interpolate(tmp_1, size=(80, 80), mode='bilinear')
        tmp_3 = torch.cat((tmp_2, in_1), 1)
        tmp_2 = None
        return (tmp_1, tmp_3)