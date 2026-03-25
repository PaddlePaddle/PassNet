import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        topk = in_0.topk(300);  in_0 = None
        getitem = topk[0]
        getitem_1 = topk[1];  topk = None
        return (getitem, getitem_1)
        