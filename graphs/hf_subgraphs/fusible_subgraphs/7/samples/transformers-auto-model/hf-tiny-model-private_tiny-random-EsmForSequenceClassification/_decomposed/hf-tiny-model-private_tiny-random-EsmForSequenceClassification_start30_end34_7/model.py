import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0 + in_1
        tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
        tmp_0 = None
        tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
        tmp_1 = None
        tmp_3 = tmp_2.to(torch.float32)
        tmp_2 = None
        return (tmp_3,)