import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0 * 0.05103103630798288
        tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
        tmp_0 = None
        tmp_2 = torch.nn.functional.dropout(tmp_1, 0.0, False, False)
        tmp_1 = None
        return (tmp_2,)