import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.cat((in_0, in_2), dim=-1)
        tmp_1 = in_1.cos()
        tmp_2 = in_1.sin()
        tmp_3 = torch.cat((tmp_1, tmp_2), dim=-1)
        tmp_1 = tmp_2 = None
        tmp_4 = torch.stack((tmp_0, tmp_3), dim=-1)
        tmp_0 = tmp_3 = None
        tmp_5 = tmp_4.transpose(-1, -2)
        tmp_4 = None
        return (tmp_5,)