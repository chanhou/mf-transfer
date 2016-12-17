function [test_RMSE ] = TPCF_new(R, R_val, RT, d, ind_u_train, ind_v_train, RN, ind_u_RN, ind_v_RN, alpha, beta, flag)
test_RMSE = []; 
alpha_u = 0.00; alpha_v = 0.00;
[test_RMSE] = learn(R, R_val, RT, d, ind_u_train, ind_v_train, RN, ind_u_RN, ind_v_RN, alpha, beta, flag)
end

function [test_RMSE ] = learn(R, R_val, RT, d, ind_u_train, ind_v_train, RN, ind_u_RN, ind_v_RN, alpha, beta, flag)
randn('state', 0);
rand('state', 0);
test_RMSE = [];
n_user = 500%max(R(:,1)) - min(R(:,1)) + 1;
n_item = 1000%max(R(:,2)) - min(R(:,2)) + 1;
aux_n_user = 500%max(RN(:,1)) - min(RN(:,1)) + 1;
aux_n_item = 500%max(RN(:,2)) - min(RN(:,2)) + 1;

mean_r = mean(R(:,3));
mean_r_aux = mean(RN(:,3));

[n_user,n_item]
l_u = 0.1*randn(n_user,d);
l_v = 0.1*randn(n_item,d);
g_u  = rand(n_user , d) ;
g_v  = rand(n_item , d) ;

l_u_aux = 0.1*randn(aux_n_user,d);
l_v_aux = 0.1*randn(aux_n_item,d);

g_u_aux  = 1*rand(aux_n_user , d) ;
g_v_aux  = 1*rand(aux_n_item , d) ;

m_u =  mean(l_u) + (alpha)*mean(l_u_aux);
m_v =  mean(l_v)  + (alpha) *mean(l_v_aux);

cov_u = ((l_u - repmat(m_u,size(l_u,1),1))' *  (l_u - repmat(m_u,size(l_u,1),1)) + diag(sum(g_u,1)'))./n_user ;
cov_v = ((l_v - repmat(m_v,size(l_v,1),1))' *  (l_v - repmat(m_v,size(l_v,1),1)) + diag(sum(g_v,1)'))./n_item;
cov_u = cov_u + (alpha) * ((l_u_aux - repmat(m_u,size(l_u_aux,1),1))' *  (l_u_aux - repmat(m_u,size(l_u_aux,1),1)) + diag(sum(g_u_aux,1)'))./aux_n_user;
cov_v = cov_v + (alpha) * ((l_v_aux - repmat(m_v,size(l_v_aux,1),1))' *  (l_v_aux - repmat(m_v,size(l_v_aux,1),1)) + diag(sum(g_v_aux,1)'))./aux_n_item;

u = R(:,1); v = R(:,2); r = R(:,3) - mean_r;
if flag == 0
    sigma = (sum(r.^2)  + sum(sum((l_u(u,:).*g_v(v,:)).*l_u(u,:))) + sum(sum((l_v(v,:).*g_u(u,:)).*l_v(v,:)))...
        - 2 * sum(r.*sum(l_u(u,:).*l_v(v,:),2)) + sum(sum((l_u(u,:).*l_v(v,:)),2).^2) + sum(sum(g_u(u,:).*g_v(v,:))))./size(R,1);
else
    sigma = 1;
    mean_r = 2.5;
end

u = RN(:,1); v = RN(:,2); r = RN(:,3) - mean_r_aux;
sigma2 = (sum(r.^2)  + sum(sum((l_u_aux(u,:).*g_v_aux(v,:)).*l_u_aux(u,:))) + sum(sum((l_v_aux(v,:).*g_u_aux(u,:)).*l_v_aux(v,:)))...
    - 2 * sum(r.*sum(l_u_aux(u,:).*l_v_aux(v,:),2)) + sum(sum((l_u_aux(u,:).*l_v_aux(v,:)),2).^2) + sum(sum(g_u_aux(u,:).*g_v_aux(v,:))))./size(RN,1);


fff = fopen('test3_result_final.txt','a');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
max_iter = 300;

best_val_rmse = 1000;
best_l_u = l_u;
best_l_v = l_v;
prev_rmse = 1000;
[rmse]  = predict(R_val, l_u , l_v,mean_r)
[rmse1]  = predict(R, l_u , l_v,mean_r)
for iter = 1 : max_iter
    inv_u = inv(cov_u); inv_u_mul_m_u = inv_u * m_u';
    inv_v = inv(cov_v); inv_v_mul_m_v = inv_v * m_v';
    %%% e_step
    
    obj = 0;
    for e_step = 1 : 10      
        for k =1 : 2
            fprintf('iter %d/%d e_step %d/20...\n',iter,max_iter,(e_step-1) * 2 + k);
            parfor i  = 1 : n_user
                if flag == 0
                    ind = ind_u_train{i};
                    r = R(ind,3) - mean_r;
                    v_ind = R(ind  , 2);
                    temp  = 1./sigma .*(l_v(v_ind,:)'*l_v(v_ind,:) + diag(sum(g_v(v_ind,:),1))');
                    temp1 = 1./sigma .* sum(repmat(r,1,d).*l_v(v_ind,:),1)';
                    temp2 =  1./sigma .* sum(l_v(v_ind,:).^2 + g_v(v_ind,:),1);
                else
                    temp = 0; temp1 = 0; temp2 = 0;
                end
                l_u(i,:) = inv(inv_u + temp) * (inv_u_mul_m_u + temp1) ;
                g_u(i,:) = 1./(diag(inv_u)' + temp2 );
            end
            parfor i  = 1 : n_item
                if flag == 0
                    ind = ind_v_train{i};
                    r = R(ind,3) - mean_r;
                    u_ind = R(ind  , 1);
                    temp = 1./sigma .*(l_u(u_ind,:)'*l_u(u_ind,:) + diag(sum(g_u(u_ind,:),1))');
                    temp1 =1./sigma .* sum(repmat(r,1,d).*l_u(u_ind,:),1)';
                    temp2 =1./sigma .* sum(l_u(u_ind,:).^2 + g_u(u_ind,:),1);
                else
                    temp = 0 ;temp1 = 0; temp2= 0 ;
                end
                l_v(i,:) = inv(inv_v + temp ) * (inv_v_mul_m_v + temp1) ;
                g_v(i,:) = 1./(diag(inv_v)' + temp2);
                
            end
            parfor i  = 1 : aux_n_user
                ind = ind_u_RN{i};
                r = RN(ind,3) - mean_r_aux;
                v_ind = RN(ind  , 2);
                temp  = 1./sigma2 .*(l_v_aux(v_ind,:)'*l_v_aux(v_ind,:) + diag(sum(g_v_aux(v_ind,:),1))');
                temp1 = 1./sigma2 .* sum(repmat(r,1,d).*l_v_aux(v_ind,:),1)';
                temp2 =  1./sigma2 .* sum(l_v_aux(v_ind,:).^2 + g_v_aux(v_ind,:),1);

                l_u_aux(i,:) = inv(inv_u +  temp ) *(inv_u_mul_m_u + temp1) ;
                g_u_aux(i,:) = 1./(diag(inv_u)' +temp2 );
            end
            parfor i  = 1 : aux_n_item
                ind = ind_v_RN{i};
                r = RN(ind,3) - mean_r_aux;
                u_ind = RN(ind  , 1);
                temp = 1./sigma2 .*(l_u_aux(u_ind,:)'*l_u_aux(u_ind,:) + diag(sum(g_u_aux(u_ind,:),1))');
                temp1 =1./sigma2 .* sum(repmat(r,1,d).*l_u_aux(u_ind,:),1)';
                temp2 =1./sigma2 .* sum(l_u_aux(u_ind,:).^2 + g_u_aux(u_ind,:),1);
                l_v_aux(i,:) = inv(inv_v + temp ) * (inv_v_mul_m_v + temp1) ;
                g_v_aux(i,:) = 1./(diag(inv_v )' + temp2);
                
            end
            
        end
    end
    %%%% m_step
    fprintf('variational M-step...\n');
    m_u =  (sum(l_u) + (alpha)*sum(l_u_aux))./(n_user + alpha * aux_n_user);
    m_v =  (sum(l_v)  + (alpha) *sum(l_v_aux))./(n_item + alpha * aux_n_item);
    
    
    temp_u = ((l_u - repmat(m_u,size(l_u,1),1))' *  (l_u - repmat(m_u,size(l_u,1),1)) + diag(sum(g_u,1)')) ;
    temp_v = ((l_v - repmat(m_v,size(l_v,1),1))' *  (l_v - repmat(m_v,size(l_v,1),1)) + diag(sum(g_v,1)'));
    temp_u1 = (alpha) * ((l_u_aux - repmat(m_u,size(l_u_aux,1),1))' *  (l_u_aux - repmat(m_u,size(l_u_aux,1),1)) + diag(sum(g_u_aux,1)'));
    temp_v1 = (alpha) * ((l_v_aux - repmat(m_v,size(l_v_aux,1),1))' *  (l_v_aux - repmat(m_v,size(l_v_aux,1),1)) + diag(sum(g_v_aux,1)'));
    cov_u = (temp_u + temp_u1)./(n_user + alpha *aux_n_user);
    cov_v = (temp_v + temp_v1)./(n_item + alpha *aux_n_item);
    [rmse]  = predict(R_val, l_u , l_v,mean_r);
    [rmse1]  = predict(R, l_u , l_v,mean_r);
    if rmse < best_val_rmse
        best_val_rmse = rmse;
        best_l_u = l_u;
        best_l_v = l_v;
    end
    if rmse > prev_rmse
        fprintf('early stopped !!!\n');
        save('./final_model/test3','l_u','l_v','mean_r','-v7.3');
        break;
    end
    prev_rmse = rmse;
    fprintf(fff,'%d,%.7f,%d,%.6f,%.6f,%.6f\n' , d,alpha,iter,rmse, rmse1, best_val_rmse);
    fprintf('val RMSE = %.5f , train RMSE = %.5f, best val RMSE = %.5f\n' , rmse, rmse1, best_val_rmse);
    u = R(:,1); v = R(:,2); r = R(:,3) - mean_r;
    sigma = (sum(r.^2)  + sum(sum((l_u(u,:).*g_v(v,:)).*l_u(u,:))) + sum(sum((l_v(v,:).*g_u(u,:)).*l_v(v,:)))...
        - 2 * sum(r.*sum(l_u(u,:).*l_v(v,:),2)) + sum(sum((l_u(u,:).*l_v(v,:)),2).^2) + sum(sum(g_u(u,:).*g_v(v,:))))./size(R,1);
    u = RN(:,1); v = RN(:,2); r = RN(:,3) - mean_r_aux;
    sigma2 = (sum(r.^2)  + sum(sum((l_u_aux(u,:).*g_v_aux(v,:)).*l_u_aux(u,:))) + sum(sum((l_v_aux(v,:).*g_u_aux(u,:)).*l_v_aux(v,:)))...
        - 2 * sum(r.*sum(l_u_aux(u,:).*l_v_aux(v,:),2)) + sum(sum((l_u_aux(u,:).*l_v_aux(v,:)),2).^2) + sum(sum(g_u_aux(u,:).*g_v_aux(v,:))))./size(RN,1);
    if flag == 1
        sigma = 1;
    end
    
end
[test_RMSE ]  = predict(RT, best_l_u , best_l_v,mean_r);
end

function acc = predict_lr(rating , l_u,l_v)
u = rating(:,1);
v = rating(:,2);
r = rating(:,3);
pred = sigm(sum(l_u(u,:).*l_v(v,:) , 2)) ;
pred = pred >= 0.5;
acc = sum(r==pred)/size(r,1);
end

function [rmse]  = predict(rating , l_u , l_v,mean_r)
u = rating(:,1);
v = rating(:,2);
r = rating(:,3);
pred = sum(l_u(u,:).*l_v(v,:) , 2) + mean_r;
pred(pred > 5) = 5;
pred(pred < 1) = 1;
rmse = sqrt(sum((pred - r).^2)./size(u,1));
end

function x = phi(x)
x = (1./(2.*x)) .* (sigm(x) - 0.5);
x(find(isnan(x))) = 1/8;
end

function x = sigm(x)
x = 1./(1+exp(-x));
end
