
all: shuffle-test1 shuffle-test2 shuffle-test3

shuffle-test1:
	shuf -n 165038 test1/train.txt > test1/train_valid.txt # 20%
	shuf -n 330000 test1/source.txt > test1/source_valid.txt # 20%
	python erase.py test1
	python to_matlab.py test1

shuffle-test2:
	shuf -n 57103 test2/train.txt > test2/train_valid.txt # 20%
	shuf -n 230760 test2/source.txt > test2/source_valid.txt # 20%
	python erase.py test2
	python to_matlab.py test2

shuffle-test3:
	shuf -n 605 test3/train.txt > test3/train_valid.txt # 10%
	shuf -n 842 test3/source.txt > test3/source_valid.txt # 10%
	python erase.py test3
	python to_matlab.py test3

test1-baseline: # 0.1789
	# ./libmf-2.01/mf-train -l2 0.01 -f 0 -k 128 -t 100 -r 0.1 -s 4 -v 5 --nmf test1/train.txt test1/baseline/baseline1
	./libmf-2.01/mf-train -l2 0.01 -f 0 -k 256 -t 23 -r 0.1 -s 4 --nmf -p test1/valid.txt test1/train_val.txt test1/baseline/baseline1

test2-baseline: # 0.1785
	# ./libmf-2.01/mf-train -l2 0.01 -f 0 -k 128 -t 100 -r 0.1 -s 4 -v 5 --nmf test2/train.txt test2/baseline/baseline1
	./libmf-2.01/mf-train -l2 0.01 -f 0 -k 256 -t 18 -r 0.1 -s 4 --nmf -p test2/valid.txt test2/train_val.txt test2/baseline/baseline1

test3-baseline: # 1.3906
	# ./libmf-2.01/mf-train -l2 0.01 -f 0 -k 128 -t 100 -r 0.1 -s 4 -v 5 --nmf test3/train.txt test3/baseline/baseline1
	./libmf-2.01/mf-train -l2 0.1 -f 0 -k 256 -t 27 -r 0.1 -s 4 --nmf -p test3/valid.txt test3/train_val.txt test3/baseline/baseline1
