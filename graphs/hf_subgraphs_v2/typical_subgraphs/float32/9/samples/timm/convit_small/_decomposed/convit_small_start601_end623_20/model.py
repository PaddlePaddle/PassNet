import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, in_0, in_1, in_2):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = torch.nn.functional.gelu(in_2, approximate='none')
        tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
        tmp_5 = None
        tmp_7 = torch.nn.functional.linear(tmp_6, tmp_4, tmp_3)
        tmp_6 = tmp_4 = tmp_3 = None
        tmp_8 = torch.nn.functional.dropout(tmp_7, 0.0, False, False)
        tmp_7 = None
        tmp_9 = in_1 + tmp_8
        tmp_8 = None
        tmp_10 = torch.cat((in_0, tmp_9), dim=1)
        tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (432,), tmp_2, tmp_1, 1e-06)
        tmp_2 = tmp_1 = None
        tmp_12 = torch.nn.functional.linear(tmp_11, tmp_0, None)
        tmp_11 = tmp_0 = None
        tmp_13 = tmp_12.reshape(1, 197, 3, 9, 48)
        tmp_12 = None
        tmp_14 = tmp_13.permute(2, 0, 3, 1, 4)
        tmp_13 = None
        tmp_15 = tmp_14.unbind(0)
        tmp_14 = None
        tmp_16 = tmp_15[0]
        tmp_17 = tmp_15[1]
        tmp_18 = tmp_15[2]
        tmp_15 = None
        tmp_19 = tmp_17.transpose(-2, -1)
        tmp_17 = None
        tmp_20 = tmp_16 @ tmp_19
        tmp_16 = tmp_19 = None
        tmp_21 = tmp_20 * 0.14433756729740643
        tmp_20 = None
        tmp_22 = tmp_21.softmax(dim=-1)
        tmp_21 = None
        tmp_23 = torch.nn.functional.dropout(tmp_22, 0.0, False, False)
        tmp_22 = None
        tmp_24 = tmp_23 @ tmp_18
        tmp_23 = tmp_18 = None
        tmp_25 = tmp_24.transpose(1, 2)
        tmp_24 = None
        tmp_26 = tmp_25.reshape(1, 197, 432)
        tmp_25 = None
        return (tmp_10, tmp_26)