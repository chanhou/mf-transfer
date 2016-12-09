function [U2,B,V2,r2] = model1(dataset,r2_train,source,r2_valid,ITER,lr,lrB,tol,lamdaU,lamdaI,lamdaB,K,C)

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
    %{
    U1 = max(ones(N, K)*(1/K) + (rand(N, K)-ones(N, K)*0.5)*0.01,0);
    U2 = max(ones(N, K)*(1/K) + (rand(N, K)-ones(N, K)*0.5)*0.01,0);
    V1 = max(ones(M, C)*(1/C) + (rand(M, C)-ones(M, C)*0.5)*0.01,0);
    V2 = max(ones(M, C)*(1/C) + (rand(M, C)-ones(M, C)*0.5)*0.01,0);
    B = max(ones(K, C)*(1/C) + (rand(K, C)-ones(K, C)*0.5)*0.01,0);
    %}
    %{
    addpath('/data/clustering.project/clustering');
    fprintf('initialize...'); tic;
    U1 = full(ind2vec(litekmeans(r1, K)')') + 0.2;
    fprintf('U1..');
    U2 = full(ind2vec(litekmeans(r2, K)')') + 0.2;
    fprintf('U2..');
    V1 = (full(ind2vec(litekmeans(r1', C)')') + 0.2)';
    fprintf('V1..');
    V2 = (full(ind2vec(litekmeans(r2', C)')') + 0.2)';
    fprintf('V2..');
    B = (U1'*r1*V1' + U2'*r2*V2')./2;
    fprintf('%f\n',toc);
    V1 = V1';
    V2 = V2';
    save(strcat('./datainit.',num2str(K),'.mat'),'U1','U2','V1','V2','B','-v7.3');
    %}
    load(strcat('./datainit.',num2str(K),'.mat'));

    pred2 = U2*B*V2';
    pred2 = max(pred2,0); pred2 = min(pred2,1);
    train_loss = calRMSE(r2,pred2);
    val_loss = calRMSE(val,pred2);
    fprintf('Iter: %d, train loss: %f, val_loss: %f\n',iter,train_loss,val_loss); 
 
    [row1,col1,value1] = find(r1);
    [row2,col2,value2] = find(r2);
    while 1,
        count = 0;
        % random shuffle index
        p = randperm(length(row1));
        row1 = row1(p); col1 = col1(p);
        p = randperm(length(row2));
        row2 = row2(p); col2 = col2(p); 

        tic; 
        for it=1:length(row1)*length(row2)
        
            %[u1,idx] = datasample(row1,1,'Replace',false);
            %i1 = col1(idx);
            %[u2,idx] = datasample(row2,1,'Replace',false);
            %i2 = col2(idx);
            u1 = row1(it); i1 = col1(it);
            u2 = row2(it); i2 = col2(it);

            pred1 = U1(u1,:)*B*V1(i1,:)';
            pred2 = U2(u2,:)*B*V2(i2,:)';
            pred1 = max(pred1,0); pred1 = min(pred1,1);
            pred2 = max(pred2,0); pred2 = min(pred2,1);

            
            %{
            if isnan(e1), 
                fprintf('%d,%d\n',u1,i1);
                disp(pred1);
                disp(U1(u1,:));
                disp(V1(i1,:));
                disp(B);
                error('e1 nan!'); 
            end
            %}
            e1 = r1(u1,i1) - pred1;
            U1(u1,:) = U1(u1,:) + lr*(e1*(B*V1(i1,:)')' - lamdaU*U1(u1,:) - lamdaU*(U1(u1,:)*ones(K,1)*ones(K,1)' - ones(1,K)));
            V1(i1,:) = V1(i1,:) + lr*(e1*(U1(u1,:)*B) - lamdaI*V1(i1,:) - lamdaI*(V1(i1,:)*ones(C,1)*ones(C,1)' - ones(1,C)) );
            %B = B + lrB*( e1*U1(u1,:)'*V1(i1,:) - lamdaB*B);

            e2 = r2(u2,i2) - pred2;
            U2(u2,:) = U2(u2,:) + lr*(e2*(B*V2(i2,:)')' - lamdaU*U2(u2,:) - lamdaU*(U2(u2,:)*ones(K,1)*ones(K,1)' - ones(1,K)) );
            V2(i2,:) = V2(i2,:) + lr*(e2*(U2(u2,:)*B) - lamdaI*V2(i2,:) - lamdaI*(V2(i2,:)*ones(C,1)*ones(C,1)' - ones(1,C)));
            %B = B + lrB*( e2*U2(u2,:)'*V2(i2,:) - lamdaB*B);

            %if isnan(e2), disp(pred2);error('e2 nan!'); end

            %B = B + lrB*( e1*U1(u1,:)'*V1(i1,:) + e2*U2(u2,:)'*V2(i2,:) - lamdaB*B - lamdaB*(B*ones(C,1)*ones(C,1)' - ones(K,1)*ones(C,1)'));
            B = B + lrB*( e1*U1(u1,:)'*V1(i1,:) + e2*U2(u2,:)'*V2(i2,:) - lamdaB*B);
            %{
            lll = length(B(B>100));
            if lll>0,
                disp(lll);
            end
            %}

            %
            U1 = max(U1,0);U2 = max(U2,0);
            V1 = max(V1,0);V2 = max(V2,0);
            B = max(B,0);
            %}

            if mod(count,50)==0,
                %fprintf('%f, %f, %f\n',e1,e2,norm(B,'fro'));
                validate();
            end
            count = count + 1;
        end

        %B = B + lr*( B_tmp - lamdaB*B);
                      
        validate();

        iter = iter + 1;
        if iter >= ITER,
            break;
        end
    end

function []=validate()
    pred2 = U2*B*V2';
    pred2 = max(pred2,0); pred2 = min(pred2,1);
    train_loss = calRMSE(r2,pred2);
    val_loss = calRMSE(val,pred2);
    fprintf('Iter: %d-%d, train loss: %f, val_loss: %f, time: %f\n',iter,count,train_loss,val_loss,toc); 

end


end
