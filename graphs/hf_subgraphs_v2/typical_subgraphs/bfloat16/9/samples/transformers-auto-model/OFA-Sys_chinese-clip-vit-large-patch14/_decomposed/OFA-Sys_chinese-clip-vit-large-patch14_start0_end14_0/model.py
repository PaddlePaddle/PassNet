import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor):
        tmp_11 = in_0.to(dtype = torch.float32);  in_0 = None
        to = tmp_11.to(torch.bfloat16);  tmp_11 = None
        conv2d = torch.conv2d(to, w_1, None, (14, 14), (0, 0), (1, 1), 1);  to = w_1 = None
        tmp_13 = conv2d.flatten(2);  conv2d = None
        tmp_14 = tmp_13.transpose(1, 2);  tmp_13 = None
        tmp_15 = w_3.expand(1, 1, -1);  w_3 = None
        tmp_16 = torch.cat([tmp_15, tmp_14], dim = 1);  tmp_15 = tmp_14 = None
        tmp_17 = torch.nn.functional.embedding(w_0, w_2, None, None, 2.0, False, False);  w_0 = w_2 = None
        tmp_18 = tmp_16 + tmp_17;  tmp_16 = tmp_17 = None
        tmp_19 = torch.nn.functional.layer_norm(tmp_18, (1024,), w_9, w_8, 1e-05);  tmp_18 = w_9 = w_8 = None
        tmp_20 = torch.nn.functional.layer_norm(tmp_19, (1024,), w_5, w_4, 1e-05);  w_5 = w_4 = None
        linear = torch.nn.functional.linear(tmp_20, w_7, w_6);  w_7 = w_6 = None
        tmp_22 = linear.view((1, 257, -1, 64));  linear = None
        tmp_23 = tmp_22.transpose(1, 2);  tmp_22 = None
        tmp_24 = tmp_23 * 0.125;  tmp_23 = None
        return (tmp_19, tmp_20, tmp_24)
        