import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4.view(16, -1, 512)
        tmp_5 = torch.nn.functional.linear(tmp_4, tmp_0, None)
        tmp_4 = tmp_0 = None
        tmp_6 = torch.nn.functional.dropout(tmp_5, 0.1, False, False)
        tmp_5 = None
        tmp_7 = in_5 + tmp_6
        tmp_6 = None
        tmp_8 = tmp_7.to(torch.float32)
        tmp_9 = tmp_8.pow(2)
        tmp_8 = None
        tmp_10 = tmp_9.mean(-1, keepdim=True)
        tmp_9 = None
        tmp_11 = tmp_10 + 1e-06
        tmp_10 = None
        tmp_12 = torch.rsqrt(tmp_11)
        tmp_11 = None
        tmp_13 = tmp_7 * tmp_12
        tmp_12 = None
        tmp_14 = tmp_3 * tmp_13
        tmp_3 = tmp_13 = None
        tmp_15 = torch.nn.functional.linear(tmp_14, tmp_1, None)
        tmp_14 = tmp_1 = None
        tmp_16 = torch.nn.functional.relu(tmp_15, inplace=False)
        tmp_15 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, 0.1, False, False)
        tmp_16 = None
        tmp_18 = torch.nn.functional.linear(tmp_17, tmp_2, None)
        tmp_17 = tmp_2 = None
        tmp_19 = torch.nn.functional.dropout(tmp_18, 0.1, False, False)
        tmp_18 = None
        tmp_20 = tmp_7 + tmp_19
        tmp_7 = tmp_19 = None
        tmp_21 = tmp_20.to(torch.float32)
        tmp_22 = tmp_21.pow(2)
        tmp_21 = None
        tmp_23 = tmp_22.mean(-1, keepdim=True)
        tmp_22 = None
        tmp_24 = tmp_23 + 1e-06
        tmp_23 = None
        tmp_25 = torch.rsqrt(tmp_24)
        tmp_24 = None
        return (tmp_20, tmp_25)