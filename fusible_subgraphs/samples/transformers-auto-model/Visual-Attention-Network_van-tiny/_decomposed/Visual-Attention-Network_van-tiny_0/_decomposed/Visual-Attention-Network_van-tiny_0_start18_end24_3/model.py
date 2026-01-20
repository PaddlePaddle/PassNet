import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1, w_2):
        tmp_0 = torch.conv2d(in_1, w_1, w_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = torch.nn.functional.dropout(tmp_0, 0.0, False, False)
        tmp_0 = None
        tmp_2 = w_2.unsqueeze(-1)
        tmp_3 = tmp_2.unsqueeze(-1)
        tmp_2 = None
        tmp_4 = tmp_3 * tmp_1
        tmp_3 = tmp_1 = None
        tmp_5 = in_0 + tmp_4
        tmp_4 = None
        return (tmp_5,)