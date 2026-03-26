import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        in_0 += in_1;  in_4 = in_0;  in_0 = in_1 = None
        in_4 += in_3;  tmp_0 = in_4;  in_4 = in_3 = None
        tmp_2 = torch.nn.functional.relu(tmp_0, inplace = False);  tmp_0 = None
        chunk = in_2.chunk(2, dim = 1);  in_2 = None
        tmp_4 = chunk[0]
        tmp_5 = chunk[1];  chunk = None
        chunk_1 = tmp_2.chunk(2, dim = 1);  tmp_2 = None
        tmp_7 = chunk_1[0]
        tmp_8 = chunk_1[1];  chunk_1 = None
        tmp_9 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, (32, 24))
        tmp_10 = torch.cat([tmp_9, tmp_8], dim = 1);  tmp_9 = None
        return (tmp_10, tmp_4, tmp_7, tmp_5, tmp_8)
        