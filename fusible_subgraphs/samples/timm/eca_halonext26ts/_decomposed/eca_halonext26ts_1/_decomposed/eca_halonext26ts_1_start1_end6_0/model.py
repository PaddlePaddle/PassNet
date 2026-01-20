import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0):
        tmp_0 = torch.conv2d(in_0, w_0, None, (2, 2), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0.reshape(-1, 16, 2, 4, 2, 4)
        tmp_0 = None
        tmp_2 = tmp_1.permute(0, 1, 3, 5, 2, 4)
        tmp_1 = None
        tmp_3 = tmp_2.reshape(8, 16, -1, 4)
        tmp_2 = None
        tmp_4 = tmp_3.transpose(1, 3)
        tmp_3 = None
        return (tmp_4,)