function rmse = calRMSE(r_val,r_predict)
    
    [row,col,val] = find(r_val);
    residual = r_val - r_predict;
    %
    res = [];
    for i=1:length(row),
        res = [res residual(row(i),col(i))];
    end
    rmse =  sqrt(sum(res.^2)/length(res));
    %}
    %{
    residual = residual(row,col);
    residual = reshape(residual,[1,size(residual,1)*size(residual,2)]);
    rmse =  sqrt(sum(residual.^2)/length(residual));
    %}
end
