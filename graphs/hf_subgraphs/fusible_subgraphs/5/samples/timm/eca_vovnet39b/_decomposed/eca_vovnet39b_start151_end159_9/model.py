import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_2.sigmoid()
        tmp_1 = tmp_0.view(1, -1, 1, 1)
        tmp_0 = None
        tmp_2 = tmp_1.expand_as(in_1)
        tmp_1 = None
        tmp_3 = in_1 * tmp_2
        tmp_2 = None
        tmp_4 = tmp_3 + in_0
        tmp_3 = None
        tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
        tmp_4 = None
        tmp_6 = tmp_5.flatten(1, -1)
        tmp_5 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
        tmp_6 = None
        return (tmp_7,)