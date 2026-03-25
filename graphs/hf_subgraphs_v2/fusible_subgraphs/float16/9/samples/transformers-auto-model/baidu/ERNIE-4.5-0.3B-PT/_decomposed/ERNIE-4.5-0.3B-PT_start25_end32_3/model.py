import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor):
        tmp_0 = torch.cat((in_0, in_0), dim = -1);  in_0 = None
        tmp_1 = tmp_0.cos()
        tmp_2 = tmp_1 * 1.0;  tmp_1 = None
        tmp_3 = tmp_0.sin();  tmp_0 = None
        tmp_4 = tmp_3 * 1.0;  tmp_3 = None
        _log_api_usage_once = torch._C._log_api_usage_once('python.nn_module');  _log_api_usage_once = None
        return (tmp_2, tmp_4)
        