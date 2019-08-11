import numpy as np
import operator
a = {"ca":1,"b":124,"c":24}

print(sorted(a,key=operator.itemgetter(0)))