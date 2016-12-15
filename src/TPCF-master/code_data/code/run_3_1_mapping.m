clc ; close all ; clear all;
RMSE = [];
MAE = [];
d = 10;
alpha = [ 1e-3, 1e-4, 1e-5];
beta = [1e-3, 1e-4, 1e-5];
dd = [20];

load ../../../../test1/data.mat

train_rating = train_train;
val_rating = train_valid;
test_rating = val_rating;

RN = source;
RU = load('./md3_1_mapping/RU.shuf');
RI = load('./md3_1_mapping/RI.shuf');
            
            n_user = 50000;% max(RI(:,1)) - min(train_rating(:,1)) + 1;
            n_item = 5000;% max(RU(:,2)) - min(train_rating(:,2)) + 1;
            aux_n_user = 50000;%max(RN(:,1)) - min(RN(:,1)) + 1;
            aux_n_item = 5000;%max(RN(:,2)) - min(RN(:,2)) + 1;
            for i = 1 : n_user
                if mod(i,500) == 0
                    fprintf('cache the index so that we don"t need to perform find every time... %d/%d\n',i,n_user);
                end
                ind_u_train{i} = find(train_rating(:,1) == i);
                ind_u_RU{i} = find(RU(:,1) == i);
                ind_u_RI{i} = find(RI(:,1) == i);
            end
            for i = 1 : n_item
                if mod(i,500) == 0
                    fprintf('cache the index so that we don"t need to perform find every time... %d/%d\n',i,n_item);
                end
                ind_v_train{i} = find(train_rating(:,2) == i);
                ind_v_RU{i} = find(RU(:,2) == i);
                ind_v_RI{i} = find(RI(:,2) == i);
                
            end
            for i = 1 : aux_n_user
                if mod(i,500) == 0
                    fprintf('cache the index so that we don"t need to perform find every time... %d/%d\n',i,n_user);
                end
                ind_u_RN{i} = find(RN(:,1) == i);
            end
            for i = 1 : aux_n_item
                if mod(i,500) == 0
                    fprintf('cache the index so that we don"t need to perform find every time... %d/%d\n',i,n_item);
                end
                ind_v_RN{i} = find(RN(:,2) == i);
            end
for d = dd,
    for a = 1 : numel(alpha)
        for b = 1 : numel(beta)

            flag = 0;
            [rmse ] =TPCF_3_1_mapping(train_rating , val_rating,test_rating, d ,ind_u_train , ind_v_train , RU,ind_u_RU,ind_v_RU , RI , ind_u_RI,ind_v_RI , RN,ind_u_RN,ind_v_RN,alpha(a),beta(b),flag);
            RMSE = [RMSE];
            MAE = [MAE];
        end
    end
end
