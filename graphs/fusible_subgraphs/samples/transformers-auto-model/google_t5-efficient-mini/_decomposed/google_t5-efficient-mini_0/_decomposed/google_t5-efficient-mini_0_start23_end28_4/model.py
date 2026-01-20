import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        in_1 += in_0
        tmp_0 = in_1
        tmp_1 = tmp_0.float()
        tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
        tmp_1 = None
        tmp_3 = tmp_2.type_as(tmp_0)
        tmp_2 = tmp_0 = None
        tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
        tmp_3 = None
        return (tmp_4,)