import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor):
        conv2d = torch.conv2d(in_5, in_4, None, (2, 2), (2, 2), (1, 1), 448);  in_5 = in_4 = None
        tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  conv2d = in_0 = in_1 = in_3 = in_2 = None
        tmp_7 = torch.nn.functional.relu(tmp_6, inplace = True);  tmp_6 = None
        return (tmp_7,)
        