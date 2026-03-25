import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = in_0.reshape(1, 197, 3, 12, 64)
        tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
        tmp_2 = None
        tmp_4 = tmp_3.unbind(0)
        tmp_3 = None
        tmp_5 = tmp_4[0]
        tmp_6 = tmp_4[1]
        tmp_7 = tmp_4[2]
        tmp_4 = None
        tmp_8 = torch.nn.functional.scaled_dot_product_attention(tmp_5, tmp_6, tmp_7, attn_mask=None, dropout_p=0.0)
        tmp_5 = tmp_6 = tmp_7 = None
        tmp_9 = tmp_8.transpose(1, 2)
        tmp_8 = None
        tmp_10 = tmp_9.reshape(1, 197, 768)
        tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (768,), tmp_1, tmp_0, 1e-05)
        tmp_10 = tmp_1 = tmp_0 = None
        return (tmp_11,)