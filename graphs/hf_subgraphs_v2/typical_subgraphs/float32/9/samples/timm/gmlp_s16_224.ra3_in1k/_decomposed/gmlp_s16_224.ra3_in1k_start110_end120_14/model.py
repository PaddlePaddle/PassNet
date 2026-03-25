import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = torch.nn.functional.gelu(in_0, approximate='none')
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.0, False, False)
        tmp_4 = None
        tmp_6 = tmp_5.chunk(2, dim=-1)
        tmp_5 = None
        tmp_7 = tmp_6[0]
        tmp_8 = tmp_6[1]
        tmp_6 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (768,), tmp_1, tmp_0, 1e-05)
        tmp_8 = tmp_1 = tmp_0 = None
        tmp_10 = tmp_9.transpose(-1, -2)
        tmp_9 = None
        tmp_11 = torch.nn.functional.linear(tmp_10, tmp_3, tmp_2)
        tmp_10 = tmp_3 = tmp_2 = None
        tmp_12 = tmp_11.transpose(-1, -2)
        tmp_11 = None
        tmp_13 = tmp_7 * tmp_12
        tmp_7 = tmp_12 = None
        return (tmp_13,)