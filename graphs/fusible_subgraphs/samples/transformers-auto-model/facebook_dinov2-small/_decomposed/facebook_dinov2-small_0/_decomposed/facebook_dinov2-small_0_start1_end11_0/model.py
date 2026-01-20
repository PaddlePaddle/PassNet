import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3):
        tmp_0 = torch.conv2d(in_0, w_1, w_0, (14, 14), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0.flatten(2)
        tmp_0 = None
        tmp_2 = tmp_1.transpose(1, 2)
        tmp_1 = None
        tmp_3 = w_2.expand(1, -1, -1)
        tmp_4 = torch.cat((tmp_3, tmp_2), dim=1)
        tmp_3 = tmp_2 = None
        tmp_5 = w_3[slice(None, None, None), slice(None, 1, None)]
        tmp_6 = w_3[slice(None, None, None), slice(1, None, None)]
        tmp_7 = tmp_6.reshape(1, 37, 37, 384)
        tmp_6 = None
        tmp_8 = tmp_7.permute(0, 3, 1, 2)
        tmp_7 = None
        tmp_9 = tmp_8.to(torch.float32)
        tmp_8 = None
        return (tmp_4, tmp_5, tmp_9)