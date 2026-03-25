import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_1.transpose(2, 3)
        tmp_1 = torch.matmul(in_2, tmp_0)
        tmp_0 = None
        tmp_2 = tmp_1 * 0.125
        tmp_1 = None
        tmp_3 = in_0[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 11, None)]
        tmp_4 = tmp_2 + tmp_3
        tmp_2 = tmp_3 = None
        tmp_5 = torch.nn.functional.softmax(tmp_4, dim=-1, dtype=torch.float32)
        tmp_4 = None
        tmp_6 = tmp_5.to(torch.float32)
        tmp_5 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, p=0.0, training=False)
        tmp_6 = None
        tmp_8 = torch.matmul(tmp_7, in_3)
        tmp_7 = None
        tmp_9 = tmp_8.transpose(1, 2)
        tmp_8 = None
        tmp_10 = tmp_9.contiguous()
        tmp_9 = None
        tmp_11 = tmp_10.reshape(1, 11, -1)
        tmp_10 = None
        tmp_12 = tmp_11.contiguous()
        tmp_11 = None
        return (tmp_12,)