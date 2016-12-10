import sys

valid = set()
with open(sys.argv[1]+'/train_valid.txt','r') as f:
	for line in f:
		line = line.split()
		valid.add(line[0]+'_'+line[1])

with open(sys.argv[1]+'/train.txt','r') as f:
	with open(sys.argv[1]+'/train_train.txt','w')as w:
		for line in f:
			li = line.split()
			if li[0]+'_'+li[1] not in valid:
				w.write(line)

valid = set()
with open(sys.argv[1]+'/source_valid.txt','r') as f:
	for line in f:
		line = line.split()
		valid.add(line[0]+'_'+line[1])

with open(sys.argv[1]+'/source.txt','r') as f:
	with open(sys.argv[1]+'/source_train.txt','w')as w:
		for line in f:
			li = line.split()
			if li[0]+'_'+li[1] not in valid:
				w.write(line)
