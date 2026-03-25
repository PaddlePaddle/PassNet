import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, in_0, in_1):
        tmp_8 = in_1.transpose(1, 2);  in_1 = None
        tmp_9 = tmp_8.view(1, 512, 2, 2);  tmp_8 = None
        conv2d = torch.conv2d(tmp_9, w_5, w_4, (1, 1), (1, 1), (1, 1), 512);  tmp_9 = w_5 = w_4 = None
        tmp_11 = conv2d.flatten(2);  conv2d = None
        tmp_12 = tmp_11.transpose(1, 2);  tmp_11 = None
        tmp_13 = torch.nn.functional.gelu(tmp_12);  tmp_12 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, 0.1, False, False);  tmp_13 = None
        linear = torch.nn.functional.linear(tmp_14, w_3, w_2);  tmp_14 = w_3 = w_2 = None
        tmp_16 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_17 = tmp_16 + in_0;  tmp_16 = in_0 = None
        tmp_18 = torch.nn.functional.layer_norm(tmp_17, (128,), w_7, w_6, 1e-05);  tmp_17 = w_7 = w_6 = None
        tmp_19 = tmp_18.reshape(1, 2, 2, -1);  tmp_18 = None
        tmp_20 = tmp_19.permute(0, 3, 1, 2);  tmp_19 = None
        tmp_21 = tmp_20.contiguous();  tmp_20 = None
        tmp_22 = tmp_21.permute(0, 2, 3, 1);  tmp_21 = None
        tmp_23 = tmp_22.reshape(1, -1, 128);  tmp_22 = None
        tmp_24 = tmp_23.mean(dim = 1);  tmp_23 = None
        linear_1 = torch.nn.functional.linear(tmp_24, w_1, w_0);  tmp_24 = w_1 = w_0 = None
        return (linear_1,)
        