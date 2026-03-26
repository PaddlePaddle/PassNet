import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor):
        tmp_5 = torch.sigmoid(in_5);  in_5 = None
        tmp_6 = tmp_5.view(128, 128, 1, 1);  tmp_5 = None
        tmp_7 = in_6 * tmp_6;  in_6 = tmp_6 = None
        tmp_8 = in_7 + tmp_7;  in_7 = tmp_7 = None
        tmp_9 = torch.cat([in_8, tmp_8], 1);  in_8 = tmp_8 = None
        tmp_10 = torch.nn.functional.batch_norm(tmp_9, in_0, in_1, in_3, in_2, False, 0.1, 0.001);  tmp_9 = in_0 = in_1 = in_3 = in_2 = None
        tmp_11 = torch.prelu(tmp_10, in_4);  tmp_10 = in_4 = None
        return (tmp_11,)
        