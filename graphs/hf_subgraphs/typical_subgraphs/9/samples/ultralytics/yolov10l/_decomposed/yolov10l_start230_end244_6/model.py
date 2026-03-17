import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, in_0):
        tmp_0 = w_0
        tmp_1 = in_0.view(1, 4, 128, 400)
        tmp_2 = tmp_1.split([32, 32, 64], dim=2)
        tmp_1 = None
        tmp_3 = tmp_2[0]
        tmp_4 = tmp_2[1]
        tmp_5 = tmp_2[2]
        tmp_2 = None
        tmp_6 = tmp_3.transpose(-2, -1)
        tmp_3 = None
        tmp_7 = tmp_6 @ tmp_4
        tmp_6 = tmp_4 = None
        tmp_8 = tmp_0.item()
        tmp_0 = None
        tmp_9 = tmp_7 * tmp_8
        tmp_7 = tmp_8 = None
        tmp_10 = tmp_9.softmax(dim=-1)
        tmp_9 = None
        tmp_11 = tmp_10.transpose(-2, -1)
        tmp_10 = None
        tmp_12 = tmp_5 @ tmp_11
        tmp_11 = None
        tmp_13 = tmp_12.view(1, 256, 20, 20)
        tmp_12 = None
        tmp_14 = tmp_5.reshape(1, 256, 20, 20)
        tmp_5 = None
        return (tmp_14, tmp_13)