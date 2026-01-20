import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, w_0):
        tmp_0 = in_1.to(torch.float16)
        tmp_1 = torch.tensor(1.0, dtype=torch.float16)
        tmp_2 = tmp_1 - tmp_0
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.to(torch.bool)
        tmp_4 = tmp_2.masked_fill(tmp_3, -65504.0)
        tmp_2 = tmp_3 = None
        tmp_5 = torch.nn.functional.embedding(in_2, w_0, None, None, 2.0, False, False)
        tmp_6 = in_0 + tmp_5
        tmp_5 = None
        tmp_7 = torch.nn.functional.dropout(tmp_6, p=0.1, training=False)
        tmp_6 = None
        return (tmp_4, tmp_7)