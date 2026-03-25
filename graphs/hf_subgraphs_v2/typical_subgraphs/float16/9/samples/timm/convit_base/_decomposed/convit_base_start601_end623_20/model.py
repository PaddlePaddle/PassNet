import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, in_0, in_1, in_2):
        tmp_5 = torch.nn.functional.gelu(in_2, approximate = 'none');  in_2 = None
        tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False);  tmp_5 = None
        linear = torch.nn.functional.linear(tmp_6, w_4, w_3);  tmp_6 = w_4 = w_3 = None
        tmp_8 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_9 = in_1 + tmp_8;  in_1 = tmp_8 = None
        tmp_10 = torch.cat((in_0, tmp_9), dim = 1);  in_0 = tmp_9 = None
        tmp_11 = torch.nn.functional.layer_norm(tmp_10, (768,), w_2, w_1, 1e-06);  w_2 = w_1 = None
        linear_1 = torch.nn.functional.linear(tmp_11, w_0, None);  tmp_11 = w_0 = None
        tmp_13 = linear_1.reshape(1, 197, 3, 16, 48);  linear_1 = None
        tmp_14 = tmp_13.permute(2, 0, 3, 1, 4);  tmp_13 = None
        unbind = tmp_14.unbind(0);  tmp_14 = None
        tmp_16 = unbind[0]
        tmp_17 = unbind[1]
        tmp_18 = unbind[2];  unbind = None
        tmp_19 = tmp_17.transpose(-2, -1);  tmp_17 = None
        matmul = tmp_16 @ tmp_19;  tmp_16 = tmp_19 = None
        tmp_21 = matmul * 0.14433756729740643;  matmul = None
        tmp_22 = tmp_21.softmax(dim = -1);  tmp_21 = None
        tmp_23 = torch.nn.functional.dropout(tmp_22, 0.0, False, False);  tmp_22 = None
        matmul_1 = tmp_23 @ tmp_18;  tmp_23 = tmp_18 = None
        tmp_25 = matmul_1.transpose(1, 2);  matmul_1 = None
        tmp_26 = tmp_25.reshape(1, 197, 768);  tmp_25 = None
        return (tmp_10, tmp_26)
        