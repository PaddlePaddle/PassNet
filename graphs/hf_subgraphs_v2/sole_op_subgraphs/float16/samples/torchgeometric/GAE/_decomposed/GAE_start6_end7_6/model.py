import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor):
        tmp_1 = in_0[(slice(None, None, None), in_1)];  in_0 = in_1 = None
        sym_size_int = torch.ops.aten.sym_size.int(tmp_1, 1)
        _check_is_size = torch._check_is_size(sym_size_int);  _check_is_size = None
        ge = sym_size_int >= 0
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        le = sym_size_int <= 100;  sym_size_int = None
        _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(le, "Runtime assertion failed for expression u0 <= 100 on node 'le'");  le = _assert_scalar_default_1 = None
        return (tmp_1,)
        