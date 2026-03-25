import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        tmp_6 = in_1.transpose(1, 2);  in_1 = None
        tmp_7 = tmp_6.view(1, 512, 2, 2);  tmp_6 = None
        conv2d = torch.conv2d(tmp_7, w_3, w_2, (1, 1), (1, 1), (1, 1), 512);  tmp_7 = w_3 = w_2 = None
        tmp_9 = conv2d.flatten(2);  conv2d = None
        tmp_10 = tmp_9.transpose(1, 2);  tmp_9 = None
        tmp_11 = torch.nn.functional.gelu(tmp_10);  tmp_10 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, 0.1, False, False);  tmp_11 = None
        linear = torch.nn.functional.linear(tmp_12, w_1, w_0);  tmp_12 = w_1 = w_0 = None
        tmp_14 = torch.nn.functional.dropout(linear, 0.1, False, False);  linear = None
        tmp_15 = tmp_14 + in_0;  tmp_14 = in_0 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (128,), w_5, w_4, 1e-05);  tmp_15 = w_5 = w_4 = None
        tmp_17 = tmp_16.reshape(1, 2, 2, -1);  tmp_16 = None
        tmp_18 = tmp_17.permute(0, 3, 1, 2);  tmp_17 = None
        tmp_19 = tmp_18.contiguous();  tmp_18 = None
        return (tmp_19,)
        