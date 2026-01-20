import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0 + in_1
        tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1, dtype=torch.float32)
        tmp_0 = None
        tmp_2 = tmp_1.to(torch.float16)
        tmp_1 = None
        tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
        tmp_2 = None
        tmp_4 = tmp_3.view(2, 18, -1)
        tmp_3 = None
        return (tmp_4,)