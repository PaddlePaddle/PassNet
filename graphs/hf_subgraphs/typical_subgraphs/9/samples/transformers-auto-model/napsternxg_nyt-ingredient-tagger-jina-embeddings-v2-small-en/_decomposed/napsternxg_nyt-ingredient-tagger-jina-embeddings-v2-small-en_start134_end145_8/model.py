import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = in_1[slice(None, None, None), slice(None, None, None), slice(None, 2048, None)]
        tmp_7 = in_1[slice(None, None, None), slice(None, None, None), slice(2048, None, None)]
        tmp_8 = torch.nn.functional.gelu(tmp_6, approximate='none')
        tmp_6 = None
        tmp_9 = tmp_8 * tmp_7
        tmp_8 = tmp_7 = None
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.1, False, False)
        tmp_9 = None
        tmp_11 = torch.nn.functional.linear(tmp_10, tmp_3, tmp_2)
        tmp_10 = tmp_3 = tmp_2 = None
        tmp_12 = tmp_11 + in_0
        tmp_11 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (512,), tmp_1, tmp_0, 1e-12)
        tmp_12 = tmp_1 = tmp_0 = None
        tmp_14 = tmp_13[slice(None, None, None), 0]
        tmp_15 = torch.nn.functional.linear(tmp_14, tmp_5, tmp_4)
        tmp_14 = tmp_5 = tmp_4 = None
        tmp_16 = torch.tanh(tmp_15)
        tmp_15 = None
        return (tmp_13, tmp_16)