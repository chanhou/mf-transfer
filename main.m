
addpath('src/');

train = 'train_train_matlab.txt';
source = 'source_train_matlab.txt';
valid = 'train_valid_matlab.txt';

% test1
data = 'test1/';
K = 100;
C = 100;
tol = 1e-3;
lamdaU = 0.0001;
lamdaI = 0.0001;
lamdaB = 0.01;
iter = 30;
lr = 0.00001;
lrB = 0.0001;
[U2,B,V2,r2] = model1(data,strcat(data,train),strcat(data,source),strcat(data,valid),iter,lr,lrB,tol,lamdaU,lamdaI,lamdaB,K,C);
%[U,V,r2] = baselineSgd(data,strcat(data,train),strcat(data,source),strcat(data,valid),iter,lr,tol,lamdaU,lamdaI,lamdaB,K,C);
%[U,V,r2] = baselineProjnmf(data,strcat(data,train),strcat(data,source),strcat(data,valid),iter,lr,tol,lamdaU,lamdaI,lamdaB,K,C);


