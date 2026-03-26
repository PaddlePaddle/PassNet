import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_1 @ in_2
        tmp_1 = tmp_0 * 0.1767766952966369
        tmp_0 = None
        tmp_2 = in_0.unsqueeze(2)
        tmp_3 = tmp_1 + tmp_2
        tmp_1 = tmp_2 = None
        tmp_4 = tmp_3.softmax(dim=-1)
        tmp_3 = None
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.0, False, False)
        tmp_4 = None
        return (tmp_5,)