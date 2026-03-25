import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_1 = tmp_0.mean([2, 3])
        tmp_0 = None
        tmp_2 = torch.nn.functional.dropout(tmp_1, 0.2, False, True)
        tmp_1 = None
        return (tmp_2,)