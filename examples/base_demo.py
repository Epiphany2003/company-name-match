# -*- coding: utf-8 -*-
"""
@description: 
"""

import sys

sys.path.append('..')
import companynameparser

if __name__ == '__main__':
    company_strs = [
        "贝通信泰国有限公司",
        "天津融通信泰国际贸易有限公司",
        "中冶海外马来西亚有限公司",
        "马来西亚海外贸易咨询有限公司沈阳办事处",
    ]
    for name in company_strs:
        r = companynameparser.parse(name)
        print(r)
        # print(name, r['place'], r['brand'], r['trade'], r['suffix'])