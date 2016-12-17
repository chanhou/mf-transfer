
with open('./test1/test.txt','r') as f, open('./submit/test1_matlab.txt','w')as w:
    for line in f:
        l = line.split()
        w.write('{} {}\n'.format(int(l[0])+1,int(l[1])+1))

with open('./test2/test.txt','r') as f, open('./submit/test2_matlab.txt','w')as w:
    for line in f:
        l = line.split()
        w.write('{} {}\n'.format(int(l[0])+1,int(l[1])+1))

with open('./test3/test.txt','r') as f, open('./submit/test3_matlab.txt','w')as w:
    for line in f:
        l = line.split()
        w.write('{} {}\n'.format(int(l[0])+1,int(l[1])+1))
