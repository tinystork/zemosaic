import sys
print(sys.executable)
print('HAS PySide6?', __import__('importlib.util').find_spec('PySide6') is not None)
