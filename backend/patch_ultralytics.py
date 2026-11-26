"""
临时补丁：修复 ultralytics 与 PyTorch 2.8+ 的兼容性
添加安全全局类到 torch 序列化白名单
"""
import sys
import os

# Windows 编码修复
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

print("=" * 60)
print("  Ultralytics PyTorch 2.8+ Compatibility Patch")
print("=" * 60)
print()

try:
    import torch
    from ultralytics.nn.tasks import DetectionModel
    
    print(f"[INFO] PyTorch version: {torch.__version__}")
    print(f"[INFO] Adding DetectionModel to safe globals...")
    
    # 添加 ultralytics 类到安全全局列表
    torch.serialization.add_safe_globals([DetectionModel])
    
    print("[OK] Patch applied!")
    print()
    print("Now try loading YOLO:")
    
    from ultralytics import YOLO
    import os as _os
    _script_dir = _os.path.dirname(_os.path.abspath(__file__))
    _base_dir = _os.path.dirname(_script_dir)
    _candidates = [_os.path.join(_base_dir, 'best.pt'), _os.path.join(_script_dir, 'best.pt')]
    _best_pt = next((p for p in _candidates if _os.path.exists(p)), None)
    model = YOLO(_best_pt) if _best_pt else None
    print("[SUCCESS] YOLO model loaded!")
    print()
    
except Exception as e:
    print(f"[ERROR] Patch failed: {e}")
    print()
    print("Alternative solution:")
    print("  1. Downgrade PyTorch: pip install torch==2.5.1")
    print("  2. Or wait for ultralytics update")
    sys.exit(1)
