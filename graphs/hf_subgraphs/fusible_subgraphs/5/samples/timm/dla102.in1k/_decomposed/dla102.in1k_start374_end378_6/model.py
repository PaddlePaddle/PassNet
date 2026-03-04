import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        in_1 += in_0
        tmp_0 = in_1
        tmp_1 = torch.nn.functional.relu(tmp_0, inplace=True)
        tmp_0 = None
        tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
        tmp_1 = None
        tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
        tmp_2 = None
        return (tmp_3,)