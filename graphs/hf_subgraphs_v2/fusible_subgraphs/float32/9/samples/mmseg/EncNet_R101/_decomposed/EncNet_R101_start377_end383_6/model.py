import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.sigmoid(in_0)
        tmp_1 = tmp_0.view(1, 512, 1, 1)
        tmp_0 = None
        tmp_2 = in_1 * tmp_1
        tmp_1 = None
        tmp_3 = in_1 + tmp_2
        tmp_2 = None
        tmp_4 = torch.relu_(tmp_3)
        tmp_3 = None
        tmp_5 = torch.nn.functional.dropout2d(tmp_4, 0.1, False, False)
        tmp_4 = None
        return (tmp_5,)