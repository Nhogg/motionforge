import json
import mujoco
import platform

print(json.dumps({'python': platform.python_version(), 'mujoco':
                  mujoco.__version__}, sort_keys=True))
