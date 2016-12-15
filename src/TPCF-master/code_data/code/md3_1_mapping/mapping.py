import sys

user = {}
item = {}
count = 1
with open(sys.argv[1],'r') as u:
    for line in u:
        num = str(int(line))
        user[str(count)] = num
        count = count + 1
count = 1
with open(sys.argv[2],'r') as i:
     for line in i:
        num = str(int(line))
        item[str(count)] = num
        count = count + 1

'''
with open(sys.argv[3],'r')as f, open(sys.argv[4]+'/RU','w')as ru, open(sys.argv[4]+'/RI','w') as ri:
    for line in f:
        li = line.split()
        ru.write('{} {} {}\n'.format(user[li[0]],li[1],li[2]))
        ri.write('{} {} {}\n'.format(li[0],item[li[1]],li[2]))
'''

src_usr = {}
src_itm = {}
with open(sys.argv[3],'r')as f:
    for line in f:
        li = line.split()
        if li[0] not in src_usr:
            src_usr[li[0]] = []
        src_usr[li[0]].append([li[1],li[2]])
        if li[1] not in src_itm:
            src_itm[li[1]] = []
        src_itm[li[1]].append([li[0],li[2]])

with open(sys.argv[4]+'/RU','w')as ru, open(sys.argv[4]+'/RI','w') as ri:
    for i in range(len(user)):
        i = str(i + 1)
        if user[i] in src_usr:
            for uarr in src_usr[user[i]]:
                ru.write('{} {} {}\n'.format(i,uarr[0],uarr[1]))
    for i in range(len(item)):
        i = str(i + 1)
        if item[i] in src_itm:
            for uarr in src_itm[item[i]]:
                ri.write('{} {} {}\n'.format(uarr[0],i,uarr[1]))

