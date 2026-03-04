import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.softmax(in_0, dim=-1)
        tmp_1 = torch.nn.functional.dropout(tmp_0, p=0.0, training=False)
        tmp_0 = None
        tmp_2 = torch.bmm(tmp_1, in_1)
        tmp_1 = None
        tmp_3 = tmp_2.view(1, 8, 100, 32)
        tmp_2 = None
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        return (tmp_4,)