# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

from companynameparser.parser import Parser
from .parser import *  # 根据实际需要调整，确保能正确导出 parser 相关内容

__version__ = "0.1.8"

par = Parser()
parse = par.parse
set_custom_split_file = par.set_custom_split_file
