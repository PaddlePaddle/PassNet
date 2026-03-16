import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_7[slice(None, None, None), slice(None, None, None), slice(None, 2048, None)]
        tmp_7 = in_7[slice(None, None, None), slice(None, None, None), slice(2048, None, None)]
        tmp_8 = torch.nn.functional.gelu(tmp_6, approximate='none')
        tmp_6 = None
        tmp_9 = tmp_8 * tmp_7
        tmp_8 = tmp_7 = None
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.1, False, False)
        tmp_9 = None
        tmp_11 = torch.nn.functional.linear(tmp_10, tmp_3, tmp_2)
        tmp_10 = tmp_3 = tmp_2 = None
        tmp_12 = tmp_11 + in_6
        tmp_11 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (512,), tmp_1, tmp_0, 1e-12)
        tmp_12 = tmp_1 = tmp_0 = None
        tmp_14 = tmp_13[slice(None, None, None), 0]
        tmp_15 = torch.nn.functional.linear(tmp_14, tmp_5, tmp_4)
        tmp_14 = tmp_5 = tmp_4 = None
        tmp_16 = torch.tanh(tmp_15)
        tmp_15 = None
        return (tmp_13, tmp_16)