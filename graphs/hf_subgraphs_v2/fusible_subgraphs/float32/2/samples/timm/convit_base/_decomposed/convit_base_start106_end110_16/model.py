import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.sum(dim=-1)
        tmp_1 = tmp_0.unsqueeze(-1)
        tmp_0 = None
        in_0 /= tmp_1
        tmp_2 = in_0
        tmp_1 = None
        tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
        tmp_2 = None
        return (tmp_3,)