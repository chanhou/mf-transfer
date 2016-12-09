clc ; close all ; clear all;
RMSE = [];
MAE = [];
d = 10;
% alpha, beta notation reverse
% in this code, beta=0
alpha = [1];
beta = [0];
% on kdd2
load /tmp2/chanhou/mf-transfer/test3/data.mat
for a = 1 : numel(alpha)
    for b = 1 : numel(beta)
        n_user = max(train_train(:,1)) - min(train_train(:,1)) + 1
        n_item = max(train_train(:,2)) - min(train_train(:,2)) + 1
        aux_n_user = max(source(:,1)) - min(source(:,1)) + 1
        aux_n_item = max(source(:,2)) - min(source(:,2)) + 1
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
        [rmse ] = TPCF_new(train_train, train_valid, train_valid, d, ind_u_train , ind_v_train, source, ind_u_source, ind_v_source, alpha(a), beta(b), flag);
        RMSE = [RMSE];
        MAE = [MAE];
    end
end