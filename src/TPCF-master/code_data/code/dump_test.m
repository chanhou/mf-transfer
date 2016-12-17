
data = cellstr(char('test1','test2','test3'));

for i = 1:length(data),
    test = load(strcat('../../../../submit/',char(data(i)),'_matlab.txt'));
    load(strcat('./final_model/',char(data(i))));
    f = fopen(strcat('../../../../submit/ans_',char(data(i))),'w');

    u = test(:,1);
    v = test(:,2);
    pred = sum(l_u(u,:).*l_v(v,:) , 2) + mean_r;
    if strcmp('test3',data(i)),
        pred(pred > 5) = 5;
        pred(pred < 1) = 1;
    else
        pred(pred > 1) = 1;
        pred(pred < 0) = 0;
    end
    for ind=1:size(test,1),
        fprintf(f,'%d %d %.10f\n',test(ind,1)-1,test(ind,2)-1,pred(ind));
    end
    fclose(f);
    %clear;
end
