import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.conv2d(in_0, in_1, padding=0)
        tmp_1 = torch.cat([tmp_0], dim=0)
        tmp_0 = None
        tmp_2 = tmp_1.reshape(1, 150, 64, 64)
        tmp_1 = None
        tmp_3 = in_2.permute(0, 1, 3, 2)
        tmp_4 = tmp_3.reshape(1, 150, 512, 1, 1)
        tmp_3 = tmp_4 = None
        return (tmp_2,)