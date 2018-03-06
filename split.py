import random

imglist='./morph_setting1.list'
with open(imglist,'r') as f:
    imgs = f.readlines()
assert(len(imgs)==5475)

random.seed()
random.shuffle(imgs)
trainlist='./train.txt'
with open(trainlist,'w') as f:
    f.writelines(imgs[0:4380])

testlist='./test.txt'
with open(testlist,'w') as f:
    f.writelines(imgs[4380::])

