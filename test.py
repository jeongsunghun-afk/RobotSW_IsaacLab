import ctypes, os
from ctypes.util import find_library

print("find_library('nvidia-ml'):", find_library('nvidia-ml'))

lib = ctypes.CDLL("libnvidia-ml.so.1")
get_ver = lib.nvmlSystemGetDriverVersion
get_ver.restype = ctypes.c_int
buf = ctypes.create_string_buffer(80)

# init
if hasattr(lib, "nvmlInit_v2"):
    lib.nvmlInit_v2()
else:
    lib.nvmlInit()

ret = get_ver(buf, ctypes.c_uint(len(buf)))
print("nvmlSystemGetDriverVersion ret:", ret)
print("NVML reported driver version:", buf.value.decode())

