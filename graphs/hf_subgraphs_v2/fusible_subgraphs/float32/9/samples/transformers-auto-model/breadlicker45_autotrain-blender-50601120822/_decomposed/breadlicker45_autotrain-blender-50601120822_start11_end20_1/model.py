import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_5.to(torch.float32)
        tmp_5 = torch.tensor(1.0, dtype=torch.float32)
        tmp_6 = tmp_5 - tmp_4
        tmp_5 = tmp_4 = None
        tmp_7 = tmp_6.to(torch.bool)
        tmp_8 = tmp_6.masked_fill(tmp_7, -3.4028234663852886e+38)
        tmp_6 = tmp_7 = None
        tmp_9 = torch.nn.functional.embedding(in_4, tmp_1, None, None, 2.0, False, False)
        tmp_1 = None
        tmp_10 = tmp_0 + tmp_9
        tmp_0 = tmp_9 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, p=0.1, training=False)
        tmp_10 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (1280,), tmp_3, tmp_2, 1e-05)
        tmp_3 = tmp_2 = None
        return (tmp_8, tmp_11, tmp_12)