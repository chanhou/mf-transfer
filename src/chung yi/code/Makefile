data:=test1

# 3-2: R1: 0.182915, R2: 0.171701 
# 3-2: after mapping, 0.181202 

1-1: 
	cd ./1_MF
	g++ -O2 ./MF.cpp -o ./MF
	./MF 1 ../../../../test1/train_train_matlab.txt ../../../../test1/train_valid_matlab.txt ../../../../test1/train_valid_matlab.txt &> log_R1 &# 0.181461
	./MF 1 ../../../../test1/source_train_matlab.txt ../../../../test1/source_valid_matlab.txt ../../../../test1/source_valid_matlab.txt &> log_R2 & # 0.173383

1-2:
	cd ./1_MF
	cat train_train_matlab.txt_vali_model.txt | ./toP.sh > R1_P
	cat train_train_matlab.txt_vali_model.txt | ./toQ.sh > R1_Q
	cat source_train_matlab.txt_vali_model.txt | ./toQ.sh > R2_Q
	cat source_train_matlab.txt_vali_model.txt | ./toP.sh > R2_P

2:
	cd ./2_latentSpaceMatching_and_matchingRefinement
	[idxP idxQ] = two_block_search(load('../../../../test1/train_train_matlab.txt'), load('../../../../test1/train_valid_matlab.txt'), load('../../../../test1/train_valid_matlab.txt'), load('../1_MF/R1_P'), load('../1_MF/R1_Q'), load('../../../../test1/source_matlab.txt'), load('../1_MF/R2_P'), load('../1_MF/R2_Q'), 20, 500);
	dlmwrite('idxP', idxP);
	dlmwrite('idxQ', idxQ);

3:
	cd ./3_ratingPredictionWithMatching
	g++ -O2 combinedMF.cpp -o combinedMF
	beta=40
	./combinedMF ../../../../test1/train_train_matlab.txt ../../../../test1/train_valid_matlab.txt ../../../../test1/train_valid_matlab.txt ../../../../test1/source_matlab.txt ../2_latentSpaceMatching_and_matchingRefinement/idxP ../2_latentSpaceMatching_and_matchingRefinement/idxQ 1 $beta $beta &> log
