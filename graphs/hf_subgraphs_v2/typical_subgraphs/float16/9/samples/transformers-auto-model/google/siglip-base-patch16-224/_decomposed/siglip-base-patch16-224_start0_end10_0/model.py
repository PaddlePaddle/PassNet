import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, w_4 : torch.Tensor, w_5 : torch.Tensor, w_6 : torch.Tensor, w_7 : torch.Tensor, w_8 : torch.Tensor, w_9 : torch.Tensor, w_10 : torch.Tensor, w_11 : torch.Tensor):
        tmp_13 = in_0.to(dtype = torch.float32);  in_0 = None
        to = tmp_13.to(torch.float16);  tmp_13 = None
        conv2d = torch.conv2d(to, w_2, w_1, (16, 16), 'valid', (1, 1), 1);  to = w_2 = w_1 = None
        tmp_15 = conv2d.flatten(2);  conv2d = None
        tmp_16 = tmp_15.transpose(1, 2);  tmp_15 = None
        tmp_17 = torch.nn.functional.embedding(w_0, w_3, None, None, 2.0, False, False);  w_0 = w_3 = None
        tmp_18 = tmp_16 + tmp_17;  tmp_16 = tmp_17 = None
        tmp_19 = torch.nn.functional.layer_norm(tmp_18, (768,), w_5, w_4, 1e-06);  w_5 = w_4 = None
        linear = torch.nn.functional.linear(tmp_19, w_9, w_8);  w_9 = w_8 = None
        linear_1 = torch.nn.functional.linear(tmp_19, w_7, w_6);  w_7 = w_6 = None
        linear_2 = torch.nn.functional.linear(tmp_19, w_11, w_10);  tmp_19 = w_11 = w_10 = None
        return (tmp_18, linear_1, linear, linear_2)
        