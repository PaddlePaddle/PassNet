import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor, in_11 : torch.Tensor, in_12 : torch.Tensor):
        tmp_13 = in_0.to(dtype = torch.float32);  in_0 = None
        to = tmp_13.to(torch.float16);  tmp_13 = None
        conv2d = torch.conv2d(to, in_3, in_2, (32, 32), 'valid', (1, 1), 1);  to = in_3 = in_2 = None
        tmp_15 = conv2d.flatten(2);  conv2d = None
        tmp_16 = tmp_15.transpose(1, 2);  tmp_15 = None
        tmp_17 = torch.nn.functional.embedding(in_1, in_4, None, None, 2.0, False, False);  in_1 = in_4 = None
        tmp_18 = tmp_16 + tmp_17;  tmp_16 = tmp_17 = None
        tmp_19 = torch.nn.functional.layer_norm(tmp_18, (768,), in_6, in_5, 1e-06);  in_6 = in_5 = None
        linear = torch.nn.functional.linear(tmp_19, in_10, in_9);  in_10 = in_9 = None
        linear_1 = torch.nn.functional.linear(tmp_19, in_8, in_7);  in_8 = in_7 = None
        linear_2 = torch.nn.functional.linear(tmp_19, in_12, in_11);  tmp_19 = in_12 = in_11 = None
        return (tmp_18, linear_1, linear, linear_2)
        