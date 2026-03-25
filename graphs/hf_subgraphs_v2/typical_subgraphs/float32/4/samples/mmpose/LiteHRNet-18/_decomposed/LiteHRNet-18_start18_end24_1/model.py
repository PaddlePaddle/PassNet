import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
        tmp_1 = torch.cat((tmp_0, in_0), dim=1)
        tmp_0 = None
        tmp_2 = tmp_1.view(32, 2, 16, 64, 48)
        tmp_1 = None
        tmp_3 = torch.transpose(tmp_2, 1, 2)
        tmp_2 = None
        tmp_4 = tmp_3.contiguous()
        tmp_3 = None
        tmp_5 = tmp_4.view(32, 32, 64, 48)
        tmp_4 = None
        return (tmp_5,)