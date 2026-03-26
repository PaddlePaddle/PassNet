import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = torch.conv2d(in_1, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_0 = None
        tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=(2, 2), stride=(2, 2))
        tmp_1 = None
        tmp_3 = tmp_2.reshape(1, 128, 4, -1)
        tmp_2 = None
        return (tmp_3,)