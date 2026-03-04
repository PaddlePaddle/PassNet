import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.nn.functional.scaled_dot_product_attention(in_1, in_0, in_2, dropout_p=0.0)
        tmp_1 = tmp_0.transpose(1, 2)
        tmp_0 = None
        tmp_2 = tmp_1.reshape(1, 192, 1280)
        tmp_1 = None
        return (tmp_2,)