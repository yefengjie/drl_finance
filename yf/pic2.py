import os
from hashlib import md5

path = '/Users/yefeng/Downloads/gif'
list = []
list1 = []
# 得到所有图片的路径，加到列表list1中
root, _, files = next(os.walk(path))
for i in range(len(files)):
    line = path + '/' + str(files[i])
    list1.append(line)
    print(i)
# 计算每张图片的md5值，并将图片路径与其md5值整合到列表list中
for n in range(len(list1)):
    hash = md5()
    img = open(list1[n], 'rb')
    hash.update(img.read())
    img.close()
    list2 = [list1[n], hash.hexdigest()]
    list.append(list2)
    print(n)
m = 0
# 两两比较md5值，若相同，则删去一张图片
while m < len(list):
    t = m + 1
    while t < len(list):
        if list[m][1] == list[t][1]:
            os.remove(list[t][0])
            del list[t]
        else:
            t += 1
    m += 1
    print(m)
