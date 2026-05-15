import importlib
m = importlib.import_module('src.utils')
print('module file:', m.__file__)
print('has_load_object=', hasattr(m,'load_object'))
print('dir snippet=', [n for n in dir(m) if not n.startswith('__')][:50])
print('source\n', open(m.__file__, 'r', encoding='utf-8').read())
