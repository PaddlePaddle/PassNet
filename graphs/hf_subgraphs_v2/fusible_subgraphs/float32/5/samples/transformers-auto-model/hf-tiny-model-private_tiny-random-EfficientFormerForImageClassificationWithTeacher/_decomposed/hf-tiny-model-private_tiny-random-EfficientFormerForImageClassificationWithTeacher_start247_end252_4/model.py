import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.nn.functional.dropout(in_0, 0.1, False, False)
        tmp_1 = in_2 * tmp_0
        tmp_0 = None
        tmp_2 = in_1 + tmp_1
        tmp_1 = None
        tmp_3 = tmp_2.flatten(2)
        tmp_2 = None
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        return (tmp_4,)