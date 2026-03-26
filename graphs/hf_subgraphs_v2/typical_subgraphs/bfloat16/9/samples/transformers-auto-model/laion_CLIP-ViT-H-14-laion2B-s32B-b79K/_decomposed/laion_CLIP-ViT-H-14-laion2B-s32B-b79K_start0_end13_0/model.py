import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor, w_12 : torch.Tensor, w_13 : torch.Tensor):
        tmp_15 = in_0.to(dtype = torch.float32);  in_0 = None
        to = tmp_15.to(torch.bfloat16);  tmp_15 = None
        conv2d = torch.conv2d(to, w_1, None, (14, 14), (0, 0), (1, 1), 1);  to = w_1 = None
        tmp_17 = conv2d.flatten(2);  conv2d = None
        tmp_18 = tmp_17.transpose(1, 2);  tmp_17 = None
        tmp_19 = w_3.expand(1, 1, -1);  w_3 = None
        tmp_20 = torch.cat([tmp_19, tmp_18], dim = 1);  tmp_19 = tmp_18 = None
        tmp_21 = torch.nn.functional.embedding(w_0, w_2, None, None, 2.0, False, False);  w_0 = w_2 = None
        tmp_22 = tmp_20 + tmp_21;  tmp_20 = tmp_21 = None
        tmp_23 = torch.nn.functional.layer_norm(tmp_22, (1280,), w_13, w_12, 1e-05);  tmp_22 = w_13 = w_12 = None
        tmp_24 = torch.nn.functional.layer_norm(tmp_23, (1280,), w_5, w_4, 1e-05);  w_5 = w_4 = None
        linear = torch.nn.functional.linear(tmp_24, w_9, w_8);  w_9 = w_8 = None
        linear_1 = torch.nn.functional.linear(tmp_24, w_7, w_6);  w_7 = w_6 = None
        linear_2 = torch.nn.functional.linear(tmp_24, w_11, w_10);  tmp_24 = w_11 = w_10 = None
        return (tmp_23, linear_1, linear, linear_2)
        