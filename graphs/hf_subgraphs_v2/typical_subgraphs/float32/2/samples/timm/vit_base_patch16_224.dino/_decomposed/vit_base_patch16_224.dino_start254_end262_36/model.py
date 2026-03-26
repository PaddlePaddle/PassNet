import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.nn.functional.gelu(in_5, approximate='none')
        tmp_5 = torch.nn.functional.dropout(tmp_4, 0.0, False, False)
        tmp_4 = None
        tmp_6 = torch.nn.functional.linear(tmp_5, tmp_1, tmp_0)
        tmp_5 = tmp_1 = tmp_0 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
        tmp_6 = None
        tmp_8 = in_4 + tmp_7
        tmp_7 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (768,), tmp_3, tmp_2, 1e-06)
        tmp_8 = tmp_3 = tmp_2 = None
        tmp_10 = tmp_9[slice(None, None, None), 0]
        tmp_9 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.0, False, False)
        tmp_10 = None
        return (tmp_11,)