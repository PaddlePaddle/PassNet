import torch

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch._functorch.vmap.lazy_load_decompositions()
        tmp_0 = None
        return ()