# DMU Analytics Platform v2
import platform
from collections import namedtuple

# Monkey-patch platform to prevent WMI calls (which hang if Windows WMI service is unresponsive)
if not hasattr(platform, '_wmi_patched'):
    uname_result = namedtuple('uname_result', ['system', 'node', 'release', 'version', 'machine', 'processor'])
    platform.uname = lambda: uname_result('Windows', 'localhost', '10', '10.0.19045', 'AMD64', 'Intel64 Family 6 Model 158 Stepping 10, GenuineIntel')
    platform.win32_ver = lambda *args, **kwargs: ('10', '10.0.19045', '', '')
    platform._wmi_patched = True
