clc ; close all ; clear all;
RMSE = [];
MAE = [];
d = 10;
alpha = [1];
beta = [0.5];
for a = 1 : numel(alpha)
    for b = 1 : numel(beta)
        for iii =2 : 6
            switch iii
                case 2,  load ../data/netflix_movie_data0001.mat
                case 3,  load ../data/netflix_movie_data001.mat
                case 4,  load ../data/netflix_movie_data002.mat
                case 5,  load ../data/netflix_movie_data003.mat
                case 6,  load ../data/netflix_movie_data004.mat
            end
            fprintf('a = %d, b = %d, iii = %d\n',a,b,iii);
            
            n_user = max(RI(:,1)) - min(train_rating(:,1)) + 1;
            n_item = max(RU(:,2)) - min(train_rating(:,2)) + 1;
            aux_n_user = max(RN(:,1)) - min(RN(:,1)) + 1;
            aux_n_item = max(RN(:,2)) - min(RN(:,2)) + 1;
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
            if iii == 1
                flag = 1;
            else
                flag = 0;
            end
            [rmse ] =TPCF(train_rating , val_rating,test_rating, d ,ind_u_train , ind_v_train , RU,ind_u_RU,ind_v_RU , RI , ind_u_RI,ind_v_RI , RN,ind_u_RN,ind_v_RN,alpha(a),beta(b),flag);
            RMSE = [RMSE];
            MAE = [MAE];
        end
    end
end