import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.dropout(in_2, p=0.0, training=False)
        tmp_3 = torch.nn.functional.linear(tmp_2, tmp_1, tmp_0)
        tmp_2 = tmp_1 = tmp_0 = None
        return (tmp_3,)