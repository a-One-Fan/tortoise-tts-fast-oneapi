import torch
import intel_extension_for_pytorch as ipex

class abstract_device_class:
    def __init__(self):
        torch.use_deterministic_algorithms = True # ?
        if torch.xpu.is_available():
            self._device = torch.device("xpu")
            self._device_str = "xpu"
            memgb = ipex.xpu.get_device_properties(self._device).total_memory / (1024**3)
            print("Found an Intel GPU with {}GB memory (this is less on WSL2)".format(round(memgb, 3)))
        else:
            self._device = torch.device("cpu")
            self._device_str = "cpu"
            print("Did not find an Intel GPU, falling back to CPU")
    def get_mem(self):
        total_memory = ipex.xpu.get_device_properties(self._device).total_memory
        return total_memory, total_memory - torch.xpu.memory_allocated(self._device)
    
    def get_device(self):
        return self._device
    
    def get_device_str(self):
        return self._device_str
    
abstract_device = abstract_device_class()