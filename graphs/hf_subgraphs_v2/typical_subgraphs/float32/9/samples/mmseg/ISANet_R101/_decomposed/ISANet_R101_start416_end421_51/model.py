import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
        tmp_1 = tmp_0.view(1, 8, 8, 512, 8, 8)
        tmp_0 = None
        tmp_2 = tmp_1.permute(0, 3, 1, 4, 2, 5)
        tmp_1 = None
        tmp_3 = tmp_2.reshape(1, 512, 64, 64)
        tmp_2 = None
        tmp_4 = torch.cat([tmp_3, in_0], dim=1)
        tmp_3 = None
        return (tmp_4,)