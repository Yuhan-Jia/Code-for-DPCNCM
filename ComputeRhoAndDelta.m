function [D,rho, delta] = ComputeRhoAndDelta(X,d_c)
    %参考DP计算rho和delta
    % X: 共识嵌入矩阵
    % d_c: 截断距离
    
    n = size(X, 1); 
    rho = zeros(n, 1);  
    delta = zeros(n, 1);  
    
    D = pdist2(X, X);  % 欧几里得距离
    
    %计算每个节点的密度rho_i
    for i = 1:n
        rho(i) = sum(exp(D(i,:) / d_c));
    end
    
    % 计算每个节点的delta_i
    for i = 1:n
        % 找到所有密度大于 rho_i 的节点
        higher_density_nodes = find(rho > rho(i));
        if ~isempty(higher_density_nodes)
            distances = D(i, higher_density_nodes);
            delta(i) = min(distances);
        else
            delta(i) = max(D(i,:));  % 如果没有比 i 更高密度的点，则maxj(dij)
        end
    end
end

