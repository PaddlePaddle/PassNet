import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_2 = in_2.transpose(1, 2);  in_2 = None
        tmp_3 = tmp_2.view(8, 1280, 32, 32);  tmp_2 = None
        to = tmp_3.to(torch.float16);  tmp_3 = None
        conv2d = torch.conv2d(to, in_1, in_0, (1, 1), (1, 1), (1, 1), 1280);  to = in_1 = in_0 = None
        tmp_5 = conv2d.flatten(2);  conv2d = None
        tmp_6 = tmp_5.transpose(1, 2);  tmp_5 = None
        tmp_7 = torch.nn.functional.gelu(tmp_6);  tmp_6 = None
        tmp_8 = torch.nn.functional.dropout(tmp_7, 0.0, False, False);  tmp_7 = None
        return (tmp_8,)
        