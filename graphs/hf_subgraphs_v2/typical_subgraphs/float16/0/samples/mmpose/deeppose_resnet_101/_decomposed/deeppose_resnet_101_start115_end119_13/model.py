import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor):
        tmp_5 = torch.nn.functional.relu(in_5, inplace = True);  in_5 = None
        conv2d = torch.conv2d(tmp_5, in_4, None, (1, 1), (0, 0), (1, 1), 1);  tmp_5 = in_4 = None
        tmp_7 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  conv2d = in_0 = in_1 = in_3 = in_2 = None
        tmp_7 += in_6;  tmp_8 = tmp_7;  tmp_7 = in_6 = None
        return (tmp_8,)
        