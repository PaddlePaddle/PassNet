import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(1, 197, 3, 9, 48)
        tmp_1 = tmp_0.permute(2, 0, 3, 1, 4)
        tmp_0 = None
        tmp_2 = tmp_1.unbind(0)
        tmp_1 = None
        tmp_3 = tmp_2[0]
        tmp_4 = tmp_2[1]
        tmp_5 = tmp_2[2]
        tmp_2 = None
        tmp_6 = tmp_4.transpose(-2, -1)
        tmp_4 = None
        tmp_7 = tmp_3 @ tmp_6
        tmp_3 = tmp_6 = None
        tmp_8 = tmp_7 * 0.14433756729740643
        tmp_7 = None
        tmp_9 = tmp_8.softmax(dim=-1)
        tmp_8 = None
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.0, False, False)
        tmp_9 = None
        tmp_11 = tmp_10 @ tmp_5
        tmp_10 = tmp_5 = None
        tmp_12 = tmp_11.transpose(1, 2)
        tmp_11 = None
        tmp_13 = tmp_12.reshape(1, 197, 432)
        tmp_12 = None
        return (tmp_13,)