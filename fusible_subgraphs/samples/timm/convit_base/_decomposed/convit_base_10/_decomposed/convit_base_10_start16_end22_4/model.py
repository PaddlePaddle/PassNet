import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_1 * 0.14433756729740643
        tmp_1 = tmp_0.softmax(dim=-1)
        tmp_0 = None
        tmp_2 = torch.nn.functional.dropout(tmp_1, 0.0, False, False)
        tmp_1 = None
        tmp_3 = tmp_2 @ in_0
        tmp_2 = None
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = tmp_4.reshape(1, 197, 768)
        tmp_4 = None
        return (tmp_5,)