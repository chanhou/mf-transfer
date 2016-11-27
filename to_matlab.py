import sys

with open(sys.argv[1]+'/train_val.txt','r') as f:
    with open(sys.argv[1]+'/train_val_matlab.txt','w')as w:
        for line in f:
            li = line.split()
            w.write(str(int(li[0])+1)+' '+str(int(li[1])+1)+' '+li[2]+'\n')
        w.write('50000 5000 0\n')

with open(sys.argv[1]+'/valid.txt','r') as f:
    with open(sys.argv[1]+'/valid_matlab.txt','w') as w:
        for line in f:
            li = line.split()
            w.write(str(int(li[0])+1)+' '+str(int(li[1])+1)+' '+li[2]+'\n')
        w.write('50000 5000 0\n')

with open(sys.argv[1]+'/source.txt','r') as f:
    with open(sys.argv[1]+'/source_matlab.txt','w')as w:
        for line in f:
            li = line.split()
            w.write(str(int(li[0])+1)+' '+str(int(li[1])+1)+' '+li[2]+'\n')
        w.write('50000 5000 0\n')
