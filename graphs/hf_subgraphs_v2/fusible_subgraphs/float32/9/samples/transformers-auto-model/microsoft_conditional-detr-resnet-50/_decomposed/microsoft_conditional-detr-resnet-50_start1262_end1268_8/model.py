import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.bmm(in_0, in_1)
        tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
        tmp_0 = None
        tmp_2 = torch.nn.functional.dropout(tmp_1, p=0.0, training=False)
        tmp_1 = None
        tmp_3 = torch.bmm(tmp_2, in_2)
        tmp_2 = None
        tmp_4 = tmp_3.view(1, 8, 300, 32)
        tmp_3 = None
        tmp_5 = tmp_4.transpose(1, 2)
        tmp_4 = None
        return (tmp_5,)