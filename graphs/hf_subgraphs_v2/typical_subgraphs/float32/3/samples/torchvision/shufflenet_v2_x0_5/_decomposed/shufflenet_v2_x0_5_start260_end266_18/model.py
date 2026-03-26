import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_1 = torch.cat((in_1, tmp_0), dim=1)
        tmp_0 = None
        tmp_2 = tmp_1.view(32, 2, 96, 7, 7)
        tmp_1 = None
        tmp_3 = torch.transpose(tmp_2, 1, 2)
        tmp_2 = None
        tmp_4 = tmp_3.contiguous()
        tmp_3 = None
        tmp_5 = tmp_4.view(32, 192, 7, 7)
        tmp_4 = None
        return (tmp_5,)