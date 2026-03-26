import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor):
        tmp_11 = in_0.to(dtype = torch.float32);  in_0 = None
        to = tmp_11.to(torch.bfloat16);  tmp_11 = None
        conv2d = torch.conv2d(to, in_2, None, (14, 14), (0, 0), (1, 1), 1);  to = in_2 = None
        tmp_13 = conv2d.flatten(2);  conv2d = None
        tmp_14 = tmp_13.transpose(1, 2);  tmp_13 = None
        tmp_15 = in_4.expand(1, 1, -1);  in_4 = None
        tmp_16 = torch.cat([tmp_15, tmp_14], dim = 1);  tmp_15 = tmp_14 = None
        tmp_17 = torch.nn.functional.embedding(in_1, in_3, None, None, 2.0, False, False);  in_1 = in_3 = None
        tmp_18 = tmp_16 + tmp_17;  tmp_16 = tmp_17 = None
        tmp_19 = torch.nn.functional.layer_norm(tmp_18, (1280,), in_10, in_9, 1e-05);  tmp_18 = in_10 = in_9 = None
        tmp_20 = torch.nn.functional.layer_norm(tmp_19, (1280,), in_6, in_5, 1e-05);  in_6 = in_5 = None
        linear = torch.nn.functional.linear(tmp_20, in_8, in_7);  in_8 = in_7 = None
        tmp_22 = linear.view((1, 257, -1, 80));  linear = None
        tmp_23 = tmp_22.transpose(1, 2);  tmp_22 = None
        tmp_24 = tmp_23 * 0.11180339887498948;  tmp_23 = None
        return (tmp_19, tmp_20, tmp_24)
        