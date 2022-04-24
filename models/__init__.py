'''
实际上，如果目录中包含了__init__.py时，当用import 导入该目录时，会执行__init__.py里面的代码。

2，__init__.py的作用

我们可以在__init__.py 指定默认需要导入的模块
'''

from .vgg import *
# from vgg import *
from .dpn import *
from .lenet import *
from .senet import *
from .pnasnet import *
from .densenet import *
from .googlenet import *
from .shufflenet import *
from .resnet import *
from .resnext import *
from .preact_resnet import *
from .mobilenet import *
from .mobilenetv2 import *
from .simple_cnn import *
