import os
import sys

# find nvidia packages
import site
site_packages = site.getsitepackages()
nvidia_dir = os.path.join(site_packages[0], 'nvidia')

lib_paths = ['/usr/lib/wsl/lib', '/usr/local/cuda-12.3/lib64']
if os.path.exists(nvidia_dir):
    for sub in os.listdir(nvidia_dir):
        lib_dir = os.path.join(nvidia_dir, sub, 'lib')
        if os.path.exists(lib_dir):
            lib_paths.append(lib_dir)

current_ld = os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_LIBRARY_PATH'] = ':'.join(lib_paths) + ':' + current_ld

# try executing python again with new ld_library_path if we haven't already
if 'RELOADED' not in os.environ:
    os.environ['RELOADED'] = '1'
    os.execv(sys.executable, [sys.executable, "scripts/check_gpu.py"])
