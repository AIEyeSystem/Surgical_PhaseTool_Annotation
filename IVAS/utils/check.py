# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

import re

def has_chinese(text: str) -> bool:
    """判断字符串中是否包含汉字"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))