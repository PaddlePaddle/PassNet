import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(32, 192, 3, 16, 64)
        tmp_1 = tmp_0.permute(2, 0, 3, 1, 4)
        tmp_0 = None
        tmp_2 = tmp_1[0]
        tmp_3 = tmp_1[1]
        tmp_4 = tmp_1[2]
        tmp_1 = None
        tmp_5 = torch.nn.functional.scaled_dot_product_attention(tmp_2, tmp_3, tmp_4, dropout_p=0.0)
        tmp_2 = tmp_3 = tmp_4 = None
        tmp_6 = tmp_5.transpose(1, 2)
        tmp_5 = None
        tmp_7 = tmp_6.reshape(32, 192, 1024)
        tmp_6 = None
        return (tmp_7,)