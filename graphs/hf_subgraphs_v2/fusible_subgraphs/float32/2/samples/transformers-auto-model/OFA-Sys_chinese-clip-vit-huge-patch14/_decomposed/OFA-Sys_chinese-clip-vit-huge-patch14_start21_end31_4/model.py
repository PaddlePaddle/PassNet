import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.matmul(in_0, in_1)
        tmp_1 = tmp_0 * 1.0
        tmp_0 = None
        tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1, dtype=torch.float32)
        tmp_1 = None
        tmp_3 = tmp_2.to(torch.float32)
        tmp_2 = None
        tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.0, training=False)
        tmp_3 = None
        tmp_5 = torch.matmul(tmp_4, in_2)
        tmp_4 = None
        tmp_6 = tmp_5.transpose(1, 2)
        tmp_5 = None
        tmp_7 = tmp_6.contiguous()
        tmp_6 = None
        tmp_8 = tmp_7.reshape(1, 257, -1)
        tmp_7 = None
        tmp_9 = tmp_8.contiguous()
        tmp_8 = None
        return (tmp_9,)