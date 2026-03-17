import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, in_0):
        tmp_0 = w_0
        tmp_1 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_2 = tmp_1.view(1, 512, -1)
        tmp_3 = tmp_1.view(1, 512, -1)
        tmp_4 = tmp_3.permute(0, 2, 1)
        tmp_3 = None
        tmp_5 = torch.bmm(tmp_2, tmp_4)
        tmp_2 = tmp_4 = None
        tmp_6 = torch.max(tmp_5, -1, keepdim=True)
        tmp_7 = tmp_6[0]
        tmp_6 = None
        tmp_8 = tmp_7.expand_as(tmp_5)
        tmp_7 = None
        tmp_9 = tmp_8 - tmp_5
        tmp_8 = tmp_5 = None
        tmp_10 = torch.nn.functional.softmax(tmp_9, dim=-1)
        tmp_9 = None
        tmp_11 = tmp_1.view(1, 512, -1)
        tmp_12 = torch.bmm(tmp_10, tmp_11)
        tmp_10 = tmp_11 = None
        tmp_13 = tmp_12.view(1, 512, 64, 64)
        tmp_12 = None
        tmp_14 = tmp_13 * tmp_0
        tmp_13 = tmp_0 = None
        tmp_15 = tmp_14 + tmp_1
        tmp_14 = tmp_1 = None
        return (tmp_15,)