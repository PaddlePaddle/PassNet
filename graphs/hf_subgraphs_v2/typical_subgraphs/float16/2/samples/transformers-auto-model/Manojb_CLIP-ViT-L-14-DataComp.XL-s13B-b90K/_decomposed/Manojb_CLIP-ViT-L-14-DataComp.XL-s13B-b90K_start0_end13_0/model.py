import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor, in_13 : torch.Tensor, in_14 : torch.Tensor):
        tmp_15 = in_0.to(dtype = torch.float32);  in_0 = None
        to = tmp_15.to(torch.float16);  tmp_15 = None
        conv2d = torch.conv2d(to, in_2, None, (14, 14), (0, 0), (1, 1), 1);  to = in_2 = None
        tmp_17 = conv2d.flatten(2);  conv2d = None
        tmp_18 = tmp_17.transpose(1, 2);  tmp_17 = None
        tmp_19 = in_4.expand(1, 1, -1);  in_4 = None
        tmp_20 = torch.cat([tmp_19, tmp_18], dim = 1);  tmp_19 = tmp_18 = None
        tmp_21 = torch.nn.functional.embedding(in_1, in_3, None, None, 2.0, False, False);  in_1 = in_3 = None
        tmp_22 = tmp_20 + tmp_21;  tmp_20 = tmp_21 = None
        tmp_23 = torch.nn.functional.layer_norm(tmp_22, (1024,), in_14, in_13, 1e-05);  tmp_22 = in_14 = in_13 = None
        tmp_24 = torch.nn.functional.layer_norm(tmp_23, (1024,), in_6, in_5, 1e-05);  in_6 = in_5 = None
        linear = torch.nn.functional.linear(tmp_24, in_10, in_9);  in_10 = in_9 = None
        linear_1 = torch.nn.functional.linear(tmp_24, in_8, in_7);  in_8 = in_7 = None
        linear_2 = torch.nn.functional.linear(tmp_24, in_12, in_11);  tmp_24 = in_12 = in_11 = None
        return (tmp_23, linear_1, linear, linear_2)
        