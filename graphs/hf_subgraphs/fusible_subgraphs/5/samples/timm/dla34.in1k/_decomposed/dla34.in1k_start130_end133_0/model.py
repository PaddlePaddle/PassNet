import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
        tmp_0 = None
        tmp_2 = torch.nn.functional.dropout(tmp_1, 0.0, False, False)
        tmp_1 = None
        return (tmp_2,)