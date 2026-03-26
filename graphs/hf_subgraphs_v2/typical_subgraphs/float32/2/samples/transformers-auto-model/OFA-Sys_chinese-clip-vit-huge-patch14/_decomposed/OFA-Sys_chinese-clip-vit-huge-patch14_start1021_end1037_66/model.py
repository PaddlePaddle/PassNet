import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.view((2, 7, -1, 64))
        tmp_2 = None
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = in_4.transpose(2, 3)
        tmp_6 = torch.matmul(in_5, tmp_5)
        tmp_5 = None
        tmp_7 = tmp_6 * 0.125
        tmp_6 = None
        tmp_8 = in_3[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 7, None)]
        tmp_9 = tmp_7 + tmp_8
        tmp_7 = tmp_8 = None
        tmp_10 = torch.nn.functional.softmax(tmp_9, dim=-1, dtype=torch.float32)
        tmp_9 = None
        tmp_11 = tmp_10.to(torch.float32)
        tmp_10 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, p=0.0, training=False)
        tmp_11 = None
        tmp_13 = torch.matmul(tmp_12, tmp_4)
        tmp_12 = tmp_4 = None
        tmp_14 = tmp_13.transpose(1, 2)
        tmp_13 = None
        tmp_15 = tmp_14.contiguous()
        tmp_14 = None
        tmp_16 = tmp_15.reshape(2, 7, -1)
        tmp_15 = None
        tmp_17 = tmp_16.contiguous()
        tmp_16 = None
        return (tmp_17,)