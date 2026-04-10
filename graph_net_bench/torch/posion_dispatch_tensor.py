import torch
import torch._ops
from torch.utils._mode_utils import no_dispatch

aten = torch._ops.ops.aten


def wrap_tensor(tensor):
    """Extract the underlying torch.Tensor from PosionDispatchTensor"""
    if isinstance(tensor, torch.Tensor):
        with no_dispatch():
            return tensor.as_subclass(PosionDispatchTensor)
    return tensor


def unwrap_tensor(tensor):
    """Extract the underlying torch.Tensor from PosionDispatchTensor"""
    if isinstance(tensor, PosionDispatchTensor):
        # Get the internal tensor storage
        # Use .as_subclass(torch.Tensor) to convert back
        with no_dispatch():
            return tensor.as_subclass(torch.Tensor)
    return tensor


def wrap_args(args):
    """Recursively wrap torch.Tensor objects in args"""
    wrapped = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            wrapped.append(wrap_tensor(arg))
        else:
            wrapped.append(arg)
    return tuple(wrapped)


def unwrap_args(args):
    """Recursively unwrap PosionDispatchTensor objects in args"""
    unwrapped = []
    for arg in args:
        if isinstance(arg, PosionDispatchTensor):
            unwrapped.append(unwrap_tensor(arg))
        else:
            unwrapped.append(arg)
    return tuple(unwrapped)


class PosionDispatchTensor(torch.Tensor):
    """
    A lightweight tensor subclass that only permits specific creation operators.
    All other operations will raise a RuntimeError.
    """

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        
        # CRITICAL: Unwrap args to get raw torch.Tensor objects
        # This prevents infinite recursion
        raw_args = unwrap_args(args)
        raw_kwargs = {k: unwrap_tensor(v) if isinstance(v, PosionDispatchTensor) else v 
                      for k, v in kwargs.items()}
        
        # ============ Tensor Creation Operators ============
        # These are critical for Triton kernels to work properly
        
        if func == aten.empty_like.default:
            # raw_args[0] is a torch.Tensor, not PosionDispatchTensor
            result = torch.empty_like(raw_args[0])
            return result.as_subclass(cls)

        if func == aten.empty.memory_format:
            size = raw_args[0] if raw_args else ()
            result = torch.empty(*size, **raw_kwargs)
            return result.as_subclass(cls)

        if func == aten.zeros.default:
            result = torch.zeros(*raw_args)
            return result.as_subclass(cls)

        if func == aten.zeros_like.default:
            result = torch.zeros_like(raw_args[0])
            return result.as_subclass(cls)
        
        if func == aten.ones.default:
            result = torch.ones(*raw_args)
            return result.as_subclass(cls)

        if func == aten.ones_like.default:
            result = torch.ones_like(raw_args[0])
            return result.as_subclass(cls)

        if func == aten.full.default:
            # args[0] is size, args[1] is fill_value
            result = torch.full(*raw_args, **raw_kwargs)
            return result.as_subclass(cls)

        if func == aten.full_like.default:
            fill_value = raw_args[1] if len(raw_args) > 1 else 0
            result = torch.full_like(raw_args[0], fill_value)
            return result.as_subclass(cls)

        # Handle torch.as_tensor and internal aliasing/lifting
        if func == aten.lift_fresh.default or func == aten.alias.default or func == aten._to_copy.default:
            result = func(*raw_args, **raw_kwargs)
            return result.as_subclass(cls)

        # ============ Shape and Metadata Operators ============
        # These return native types, not tensors
        
        if func in (aten.size.default, aten.sym_size.default):
            return raw_args[0].size()
        
        if func in (aten.stride.default, aten.sym_stride.default):
            return raw_args[0].stride()
        
        if func in (aten.dim.default, aten.sym_numel.default, aten.numel.default):
            if func == aten.dim.default:
                return raw_args[0].dim()
            else:
                return raw_args[0].numel()

        # ============ Tensor Property Accessors ============
  
        if func == aten.device.default:
            return raw_args[0].device

        # ============ raise RuntimeError for All Other Operators ============
    
        raise RuntimeError(f"Unauthorized Operator ({func}) Detected: The current operation is not in the PosionDispatchTensor whitelist. "
                "Only basic creation and factory methods (empty, zeros, ones, full, as_tensor) are supported. "
                "Check the replacement_func() or the operation being called on this subclass.")

    def __repr__(self):
        """Custom representation to indicate this is a PosionDispatchTensor"""
        return f"PosionDispatchTensor({super().__repr__()})"