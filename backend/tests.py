from django.test import TestCase
import math
from views import getpoint
from views import getPictureResult
# Create your tests here.

getPictureResult()
exit()

# 二分寻找分层点
testAcc = [2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 0, 0]
head = 1
tail = 18
accThreshold = 0.01
while head < tail - 1:
    mid = math.ceil((head + tail) / 2.)
    print([head, mid, tail])
    # paramList, fr
    acc = testAcc[mid - 1]
    if acc > 1:
        acc /= 100
    if acc < accThreshold:
        tail = mid
    else:
        head = mid
# 返回
resp = {'message': "success", 'result': 'ok', 'data': 19 - tail}
print(resp)
exit()