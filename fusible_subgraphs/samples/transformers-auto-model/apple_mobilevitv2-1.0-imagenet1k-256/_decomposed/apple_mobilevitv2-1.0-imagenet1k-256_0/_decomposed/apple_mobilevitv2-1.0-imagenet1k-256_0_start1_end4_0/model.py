import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0):
        tmp_0 = torch.conv2d(in_0, w_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = torch.nn.functional.unfold(tmp_0, kernel_size=(2, 2), stride=(2, 2))
        tmp_0 = None
        tmp_2 = tmp_1.reshape(1, 128, 4, -1)
        tmp_1 = None
        return (tmp_2,)