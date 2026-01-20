import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.max(in_0, in_1)
        tmp_1 = tmp_0.view(16, 13, 13)
        tmp_0 = None
        tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
        tmp_1 = None
        tmp_3 = torch.nn.functional.dropout(tmp_2, p=0.1, training=False)
        tmp_2 = None
        return (tmp_3,)