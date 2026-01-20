import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0):
        tmp_0 = w_0[in_1]
        tmp_1 = tmp_0.view(49, 49, -1)
        tmp_0 = None
        tmp_2 = tmp_1.permute(2, 0, 1)
        tmp_1 = None
        tmp_3 = tmp_2.contiguous()
        tmp_2 = None
        tmp_4 = tmp_3.unsqueeze(0)
        tmp_3 = None
        tmp_5 = in_0 + tmp_4
        tmp_4 = None
        tmp_6 = torch.nn.functional.softmax(tmp_5, dim=-1)
        tmp_5 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
        tmp_6 = None
        return (tmp_7,)