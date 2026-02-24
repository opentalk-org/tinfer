from multiprocessing import shared_memory
import numpy as np
import torch
from typing import Any

class SharedMemoryManager:
    def __init__(self) -> None:
        self._allocated: dict[str, shared_memory.SharedMemory] = {}

    def create_shared_array(
        self, shape: tuple[int, ...], dtype: np.dtype
    ) -> tuple[shared_memory.SharedMemory, np.ndarray]:
        if shared_memory is None:
            raise RuntimeError("multiprocessing.shared_memory is not available on this platform")
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        shm = shared_memory.SharedMemory(create=True, size=size)
        array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        self._allocated[shm.name] = shm
        return shm, array

    def get_shared_array(
        self, name: str, shape: tuple[int, ...], dtype: np.dtype
    ) -> tuple[shared_memory.SharedMemory, np.ndarray]:
        if shared_memory is None:
            raise RuntimeError("multiprocessing.shared_memory is not available on this platform")
        shm = shared_memory.SharedMemory(name=name, create=False)
        array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        return shm, array

    def serialize_array(self, arr: np.ndarray | torch.Tensor) -> dict[str, Any]:
        if isinstance(arr, np.ndarray):
            shape = arr.shape
            dtype = arr.dtype
            shm, shm_array = self.create_shared_array(shape, dtype)
            np.copyto(shm_array, arr)
            return {
                "__shared_memory__": True,
                "name": shm.name,
                "shape": shape,
                "dtype": str(dtype),
                "is_tensor": False,
            } 

        elif isinstance(arr, torch.Tensor):
            if arr.is_cuda:
                arr = arr.cpu()
            arr_np = arr.numpy()
            shape = arr_np.shape
            dtype = arr_np.dtype
            shm, shm_array = self.create_shared_array(shape, dtype)
            np.copyto(shm_array, arr_np)
            return {
                "__shared_memory__": True,
                "name": shm.name,
                "shape": shape,
                "dtype": str(dtype),
                "is_tensor": True,
            }

        return 
        
    
    def deserialize_array(self, ref: dict[str, Any]) -> np.ndarray | torch.Tensor | Any:
        if not isinstance(ref, dict) or not ref.get("__shared_memory__"):
            return ref
        
        name = ref["name"]
        shape = tuple(ref["shape"])
        dtype_str = ref["dtype"]
        is_tensor = ref.get("is_tensor", False)
        
        dtype = np.dtype(dtype_str)
        shm, arr = self.get_shared_array(name, shape, dtype)
        
        arr_copy = np.copy(arr)
        shm.close()
        
        if name in self._allocated:
            try:
                self._allocated[name].unlink()
                del self._allocated[name]
            except Exception:
                pass
        
        if is_tensor and torch is not None:
            return torch.from_numpy(arr_copy)
        
        return arr_copy


    # TODO: Refactor serialize / deserialize, too much versatility, probably not needed
    def serialize_recursive(self, obj: Any, visited: set[int] | None = None) -> Any:

        if visited is None:
            visited = set()
        
        obj_id = id(obj)
        if obj_id in visited:
            return obj
        
        if isinstance(obj, (np.ndarray,)) or (torch is not None and isinstance(obj, torch.Tensor)):
            return self.serialize_array(obj)
        
        if isinstance(obj, dict):
            visited.add(obj_id)
            try:
                return {k: self.serialize_recursive(v, visited) for k, v in obj.items()}
            finally:
                visited.discard(obj_id)
        
        if isinstance(obj, (list, tuple)):
            visited.add(obj_id)
            try:
                serialized = [self.serialize_recursive(item, visited) for item in obj]
                return tuple(serialized) if isinstance(obj, tuple) else serialized
            finally:
                visited.discard(obj_id)
        
        if hasattr(obj, "__dict__"):
            visited.add(obj_id)
            try:
                obj_dict = {k: self.serialize_recursive(v, visited) for k, v in obj.__dict__.items()}
                if hasattr(obj, "__class__"):
                    return {
                        "__dataclass__": True,
                        "__class__": obj.__class__.__name__,
                        "__module__": obj.__class__.__module__,
                        "__dict__": obj_dict,
                    }
                return obj_dict
            finally:
                visited.discard(obj_id)
        
        return obj

    def cleanup(self) -> None:
        for name, shm in list(self._allocated.items()):
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
            finally:
                if name in self._allocated:
                    del self._allocated[name]

    def __del__(self) -> None:
        self.cleanup()

    def deserialize_recursive(self, obj: Any, visited: set[int] | None = None) -> Any:
        if visited is None:
            visited = set()
        
        obj_id = id(obj)
        if obj_id in visited:
            return obj
        
        if isinstance(obj, dict):
            if obj.get("__shared_memory__"):
                return self.deserialize_array(obj)
            
            if obj.get("__dataclass__"):
                from dataclasses import dataclass, fields
                import importlib
                
                module_name = obj["__module__"]
                class_name = obj["__class__"]
                obj_dict = obj["__dict__"]
                
                try:
                    module = importlib.import_module(module_name)
                    cls = getattr(module, class_name)
                    visited.add(obj_id)
                    try:
                        deserialized_dict = {k: self.deserialize_recursive(v, visited) for k, v in obj_dict.items()}
                    finally:
                        visited.discard(obj_id)
                    
                    if hasattr(cls, "__dataclass_fields__"):
                        field_names = {f.name for f in fields(cls)}
                        filtered_dict = {k: v for k, v in deserialized_dict.items() if k in field_names}
                        return cls(**filtered_dict)
                    else:
                        instance = object.__new__(cls)
                        instance.__dict__.update(deserialized_dict)
                        return instance
                except (ImportError, AttributeError):
                    visited.add(obj_id)
                    try:
                        return {k: self.deserialize_recursive(v, visited) for k, v in obj_dict.items()}
                    finally:
                        visited.discard(obj_id)
            
            visited.add(obj_id)
            try:
                return {k: self.deserialize_recursive(v, visited) for k, v in obj.items()}
            finally:
                visited.discard(obj_id)
        
        if isinstance(obj, (list, tuple)):
            visited.add(obj_id)
            try:
                deserialized = [self.deserialize_recursive(item, visited) for item in obj]
                return tuple(deserialized) if isinstance(obj, tuple) else deserialized
            finally:
                visited.discard(obj_id)
        
        return obj
