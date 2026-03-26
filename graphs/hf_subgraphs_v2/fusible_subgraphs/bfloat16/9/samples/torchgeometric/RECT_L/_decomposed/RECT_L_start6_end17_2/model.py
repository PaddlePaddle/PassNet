import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_1 = in_0[(slice(None, None, None), in_2)];  in_0 = in_2 = None
        sym_size_int = torch.ops.aten.sym_size.int(tmp_1, 1)
        _check_is_size_2 = torch._check_is_size(sym_size_int);  _check_is_size_2 = None
        ge_1 = sym_size_int >= 0
        _assert_scalar_default_2 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u0 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_default_2 = None
        le_1 = sym_size_int <= 128
        _assert_scalar_default_3 = torch.ops.aten._assert_scalar.default(le_1, "Runtime assertion failed for expression u0 <= 128 on node 'le_1'");  le_1 = _assert_scalar_default_3 = None
        _check_is_size = torch._check_is_size(sym_size_int);  _check_is_size = None
        _check_is_size_1 = torch._check_is_size(sym_size_int);  _check_is_size_1 = None
        tmp_9 = torch.cat([tmp_1, in_1], dim = 1);  tmp_1 = in_1 = None
        sym_sum = torch.sym_sum([128, sym_size_int]);  sym_size_int = None
        tmp_11 = torch.ones((sym_sum,), dtype = torch.float32, device = device(type='cuda'));  sym_sum = None
        return (tmp_9, tmp_11)
        