import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.view(32, 6, 128, 400)
        tmp_1 = tmp_0.split([32, 32, 64], dim=2)
        tmp_0 = None
        tmp_2 = tmp_1[0]
        tmp_3 = tmp_1[1]
        tmp_4 = tmp_1[2]
        tmp_1 = None
        tmp_5 = tmp_2.transpose(-2, -1)
        tmp_2 = None
        tmp_6 = tmp_5 @ tmp_3
        tmp_5 = tmp_3 = None
        tmp_7 = tmp_6 * 0.1767766952966369
        tmp_6 = None
        tmp_8 = tmp_7.softmax(dim=-1)
        tmp_7 = None
        tmp_9 = tmp_8.transpose(-2, -1)
        tmp_8 = None
        tmp_10 = tmp_4 @ tmp_9
        tmp_9 = None
        tmp_11 = tmp_10.view(32, 384, 20, 20)
        tmp_10 = None
        tmp_12 = tmp_4.reshape(32, 384, 20, 20)
        tmp_4 = None
        return (tmp_12, tmp_11)