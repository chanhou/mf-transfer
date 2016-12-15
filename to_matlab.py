import sys

if sys.argv[1]=='test1':
    train = '50000 5000 0\n'
    source = '50000 5000 0\n'
elif sys.argv[1]=='test2':
    train = '20000 2000 0\n'
    source = '30000 3000 0\n'
elif sys.argv[1]=='test3':
    train = '500 1000 0\n'
    source = '500 500 0\n'

with open(sys.argv[1]+'/train_valid.txt','r') as f:
    with open(sys.argv[1]+'/train_valid_matlab.txt','w')as w:
        for line in f:
            li = line.split()
            w.write(str(int(li[0])+1)+' '+str(int(li[1])+1)+' '+li[2]+'\n')
        #w.write('50000 5000 0\n')
        w.write(train)

with open(sys.argv[1]+'/train_train.txt','r') as f:
    with open(sys.argv[1]+'/train_train_matlab.txt','w') as w:
        for line in f:
            li = line.split()
            w.write(str(int(li[0])+1)+' '+str(int(li[1])+1)+' '+li[2]+'\n')
        #w.write('50000 5000 0\n')
        w.write(train)

with open(sys.argv[1]+'/source_valid.txt','r') as f:
    with open(sys.argv[1]+'/source_valid_matlab.txt','w')as w:
        for line in f:
            li = line.split()
            w.write(str(int(li[0])+1)+' '+str(int(li[1])+1)+' '+li[2]+'\n')
        #w.write('50000 5000 0\n')
        w.write(source)

with open(sys.argv[1]+'/source_train.txt','r') as f:
    with open(sys.argv[1]+'/source_train_matlab.txt','w') as w:
        for line in f:
            li = line.split()
            w.write(str(int(li[0])+1)+' '+str(int(li[1])+1)+' '+li[2]+'\n')
        #w.write('50000 5000 0\n')
        w.write(source)

with open(sys.argv[1]+'/source.txt','r') as f:
    with open(sys.argv[1]+'/source_matlab.txt','w') as w:
        for line in f:
            li = line.split()
            w.write(str(int(li[0])+1)+' '+str(int(li[1])+1)+' '+li[2]+'\n')
        #w.write('50000 5000 0\n')
        w.write(source)

