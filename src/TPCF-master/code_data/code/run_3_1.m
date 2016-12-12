clc ; close all ; clear all;
RMSE = [];
MAE = [];
d = 20;
% alpha, beta notation reverse
% in this code, beta=0
alpha = [1e-3,1e-4,1e-5,1e-6, 1e-7];
beta = [0];
dd = [20];

% on kdd2
%
%load /tmp2/chanhou/mf-transfer/test3/data.mat
load ../../../../test1/data.mat
%}
%{
load ../data/netflix_movie_data0001.mat
train_train = train_rating;
train_valid = val_rating;
source = RN;
%}

for d = (dd),
for a = 1 : numel(alpha)
    for b = 1 : numel(beta)
        n_user = 50000%max(train_train(:,1)) - min(train_train(:,1)) + 1
        n_item = 5000%max(train_train(:,2)) - min(train_train(:,2)) + 1
        aux_n_user = 50000%max(source(:,1)) - min(source(:,1)) + 1
        aux_n_item = 5000%max(source(:,2)) - min(source(:,2)) + 1
        for i = 1 : n_user
            if mod(i,500) == 0
                fprintf('cache the index so that we don"t need to perform find every time... %d/%d\n',i,n_user);
            end
            ind_u_train{i} = find(train_train(:,1) == i);
        end
        for i = 1 : n_item
            if mod(i,500) == 0
                fprintf('cache the index so that we don"t need to perform find every time... %d/%d\n',i,n_item);
            end
            ind_v_train{i} = find(train_train(:,2) == i);
            
        end
        for i = 1 : aux_n_user
            if mod(i,500) == 0
                fprintf('cache the index so that we don"t need to perform find every time... %d/%d\n',i,n_user);
            end
            ind_u_source{i} = find(source(:,1) == i);
        end
        for i = 1 : aux_n_item
            if mod(i,500) == 0
                fprintf('cache the index so that we don"t need to perform find every time... %d/%d\n',i,n_item);
            end
            ind_v_source{i} = find(source(:,2) == i);
        end
        flag = 0;
        [rmse ] = TPCF_3_1(train_train, train_valid, train_valid, d, ind_u_train , ind_v_train, source, ind_u_source, ind_v_source, alpha(a), beta(b), flag);
        RMSE = [RMSE];
        MAE = [MAE];
    end
end
end
