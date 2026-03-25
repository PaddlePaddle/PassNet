import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_0 = torch.nn.functional.relu(in_0, inplace = False);  in_0 = None
        chunk = in_1.chunk(2, dim = 1);  in_1 = None
        tmp_2 = chunk[0]
        tmp_3 = chunk[1];  chunk = None
        chunk_1 = in_2.chunk(2, dim = 1);  in_2 = None
        tmp_5 = chunk_1[0]
        tmp_6 = chunk_1[1];  chunk_1 = None
        chunk_2 = tmp_0.chunk(2, dim = 1);  tmp_0 = None
        tmp_8 = chunk_2[0]
        tmp_9 = chunk_2[1];  chunk_2 = None
        tmp_10 = torch.nn.functional.adaptive_avg_pool2d(tmp_3, (16, 12))
        tmp_11 = torch.nn.functional.adaptive_avg_pool2d(tmp_6, (16, 12))
        tmp_12 = torch.cat([tmp_10, tmp_11, tmp_9], dim = 1);  tmp_10 = tmp_11 = None
        return (tmp_12, tmp_8, tmp_2, tmp_5, tmp_3, tmp_6, tmp_9)
        