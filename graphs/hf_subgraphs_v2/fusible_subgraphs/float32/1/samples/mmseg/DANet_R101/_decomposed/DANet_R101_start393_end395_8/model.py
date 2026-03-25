import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_1 = torch.nn.functional.dropout2d(tmp_0, 0.1, False, False)
        return (tmp_1, tmp_0)