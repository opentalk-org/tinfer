#!/usr/bin/env python3
import re
import sys
from pathlib import Path

# TODO: fix imports, to be removed

def fix_pb2_grpc_imports(file_path: Path):
    content = file_path.read_text()
    
    content = re.sub(
        r'^import styletts_pb2 as styletts__pb2$',
        'from . import styletts_pb2 as styletts__pb2',
        content,
        flags=re.MULTILINE
    )
    
    content = re.sub(
        r'^GRPC_VERSION = grpc\.__version__$',
        'GRPC_VERSION = getattr(grpc, \'__version__\', \'unknown\')',
        content,
        flags=re.MULTILINE
    )
    
    old_version_check = r'''_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower\(GRPC_VERSION, GRPC_GENERATED_VERSION\)
except ImportError:
    _version_not_supported = True

if _version_not_supported:'''
    
    new_version_check = '''_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    if GRPC_VERSION != 'unknown':
        _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
    else:
        _version_not_supported = False
except ImportError:
    _version_not_supported = False

if _version_not_supported:'''
    
    content = re.sub(old_version_check, new_version_check, content, flags=re.MULTILINE)
    
    file_path.write_text(content)

if __name__ == '__main__':
    grpc_dir = Path(__file__).parent
    pb2_grpc_file = grpc_dir / 'styletts_pb2_grpc.py'
    
    if pb2_grpc_file.exists():
        fix_pb2_grpc_imports(pb2_grpc_file)
        print(f"Fixed imports in {pb2_grpc_file}")
    else:
        print(f"File not found: {pb2_grpc_file}", file=sys.stderr)
        sys.exit(1)


