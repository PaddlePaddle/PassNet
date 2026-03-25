import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0, in_1, in_2, in_3):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = torch.nn.functional.relu(in_3, inplace=True)
        tmp_3 = torch.cat([in_0, in_1, in_2, tmp_2], 1)
        tmp_2 = None
        tmp_4 = torch.nn.functional.adaptive_avg_pool2d(tmp_3, (1, 1))
        tmp_3 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.5, False, False)
        tmp_4 = None
        tmp_6 = torch.flatten(tmp_5, 1)
        tmp_5 = None
        tmp_7 = torch.nn.functional.linear(tmp_6, tmp_1, tmp_0)
        tmp_6 = tmp_1 = tmp_0 = None
        return (tmp_7,)