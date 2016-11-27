function [U,V,r2] = baseline(dataset,r2_train,source,r2_valid,ITER,lr,tol,lamdaU,lamdaI,lamdaB,K,C)

    rng('shuffle');
    addpath(dataset);
    tic;
    r2 = spconvert(load(r2_train));
    r1 = spconvert(load(source));
    val = spconvert(load(r2_valid));
    fprintf('load data set done, cost: %f...\n',toc);

    N = size(r2,1); % # of user
    M = size(r2,2); % # of item
    K = K; % latent dim of user
    C = C; % latent dim of item

    iter = 0;
    %initialize
    U = max(ones(N, K)*(1/K) + (rand(N, K)-ones(N, K)*0.5)*0.01,0);
    V = max(ones(M, K)*(1/K) + (rand(M, K)-ones(M, K)*0.5)*0.01,0);
    %U = rand(N,K);
    %V = rand(M,K);

    pred = U*V';
    train_loss_old = calRMSE(r2,pred);
    val_loss = calRMSE(val,pred);
    fprintf('Iter: %d, train loss: %f, val_loss: %f\n',iter,train_loss_old,val_loss); 
 
    [row,col,value] = find(r2);
    while 1,
        count = 0;
        % random shuffle index
        p = randperm(length(row));
        row = row(p); col = col(p);

        tic; 
        for ind=1:length(row),
        %for ind=1:N,
            u = row(ind); i = col(ind);
            %[u idx] = datasample(row,1);
            %i = col(idx);
            pred = U(u,:)*V(i,:)';
            e = r2(u,i) - pred;
            U(u,:) = U(u,:) + lr*(e*V(i,:) - lamdaU*U(u,:));
            V(i,:) = V(i,:) + lr*(e*U(u,:) - lamdaI*V(i,:));
            %U(U<0)=0;V(V<0)=0;
            if mod(count,1000)==0,
                [train_loss_new] = validate();
                if train_loss_new > train_loss_old,
                    lr = lr*0.1;
                    fprintf('lr change to: %f\n',lr);
                end
                train_loss_old = train_loss_new;
            end
            count = count + 1;
        end 
                      
        validate();

        iter = iter + 1;
        if iter >= ITER,
            break;
        end
    end

function [train_loss]=validate()
    pred = U*V';
    train_loss = calRMSE(r2,pred);
    val_loss = calRMSE(val,pred);
    fprintf('Iter: %d-%d, train loss: %f, val_loss: %f, time: %f\n',iter,count,train_loss,val_loss,toc); 

end


end
