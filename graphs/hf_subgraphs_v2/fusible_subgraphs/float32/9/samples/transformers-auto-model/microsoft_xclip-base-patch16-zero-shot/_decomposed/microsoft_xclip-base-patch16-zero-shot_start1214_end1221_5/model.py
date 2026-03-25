import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0 @ in_1
        tmp_1 = tmp_0 * 0.125
        tmp_0 = None
        tmp_2 = tmp_1.softmax(dim=-1)
        tmp_1 = None
        tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
        tmp_2 = None
        tmp_4 = tmp_3 @ in_2
        tmp_3 = None
        tmp_5 = tmp_4.transpose(1, 2)
        tmp_4 = None
        tmp_6 = tmp_5.reshape(1, 1, 512)
        tmp_5 = None
        return (tmp_6,)