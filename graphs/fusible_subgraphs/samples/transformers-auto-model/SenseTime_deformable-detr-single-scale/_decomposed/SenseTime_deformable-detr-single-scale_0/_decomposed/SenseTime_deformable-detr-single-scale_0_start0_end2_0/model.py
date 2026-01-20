import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.relu(in_0, inplace=False)
        tmp_1 = torch.nn.functional.dropout(tmp_0, p=0.0, training=False)
        tmp_0 = None
        return (tmp_1,)