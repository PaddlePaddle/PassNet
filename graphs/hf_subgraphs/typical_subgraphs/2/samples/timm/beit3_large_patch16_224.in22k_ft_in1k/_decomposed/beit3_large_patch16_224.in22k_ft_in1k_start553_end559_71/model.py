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
        tmp_6 = torch.nn.functional.layer_norm(tmp_5, (4096,), tmp_3, tmp_2, 1e-05)
        tmp_5 = tmp_3 = tmp_2 = None
        tmp_7 = torch.nn.functional.linear(tmp_6, tmp_1, tmp_0)
        tmp_6 = tmp_1 = tmp_0 = None
        tmp_8 = torch.nn.functional.dropout(tmp_7, 0.0, False, False)
        tmp_7 = None
        tmp_9 = in_4 + tmp_8
        tmp_8 = None
        return (tmp_9,)