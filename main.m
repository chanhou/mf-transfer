
addpath('src/');

train = 'train_val_matlab.txt';
source = 'source_matlab.txt';
valid = 'valid_matlab.txt';

% test1
data = 'test1/';
K = 128;
C = 128;
tol = 1e-3;
lamdaU = 0.01;
lamdaI = 0.01;
lamdaB = 0.05;
iter = 30;
lr = 0.1;
lrB = 0.1;
[U2,B,V2,r2] = model1(data,strcat(data,train),strcat(data,source),strcat(data,valid),iter,lr,lrB,tol,lamdaU,lamdaI,lamdaB,K,C);
%[U,V,r2] = baseline(data,strcat(data,train),strcat(data,source),strcat(data,valid),iter,lr,tol,lamdaU,lamdaI,lamdaB,K,C);

