import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0 * 1.0
        tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1, dtype=torch.float32)
        tmp_0 = None
        tmp_2 = tmp_1.to(torch.float32)
        tmp_1 = None
        tmp_3 = torch.nn.functional.dropout(tmp_2, p=0.0, training=False)
        tmp_2 = None
        tmp_4 = torch.matmul(tmp_3, in_1)
        tmp_3 = None
        tmp_5 = tmp_4.transpose(1, 2)
        tmp_4 = None
        tmp_6 = tmp_5.contiguous()
        tmp_5 = None
        tmp_7 = tmp_6.reshape(1, 257, -1)
        tmp_6 = None
        tmp_8 = tmp_7.contiguous()
        tmp_7 = None
        return (tmp_8,)