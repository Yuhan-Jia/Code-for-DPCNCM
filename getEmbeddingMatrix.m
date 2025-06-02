function X = getEmbeddingMatrix(A,S,t,k,l)
    % A是邻接矩阵，S是属性
    % t,k,l分别是拓扑、属性、共识嵌入时选择的特征向量的维数
    % 输出公式嵌入矩阵
    
    %% 拓扑信息嵌入
    nodes_degree = sum(A, 2); % 节点的度
    % 度矩阵
    D_A = diag(nodes_degree); 
    % Laplacian矩阵
    L_A = D_A - A;
    % 选择前 t 个特征向量，特征值按升序排序
    [V_A, E_A] = eig(L_A, D_A);
    [sorted_eigenvalues_A, sorted_indices_A] = sort(diag(E_A), 'ascend'); % 改为升序
    X_A = V_A(:, sorted_indices_A(2:t+1));  % 排除第一个特征向量
    
    %% 属性信息嵌入
    %删除全0列
    S(:, sum(S, 1) == 0) = [];
    S_normalized = normalize(S);  % 标准化属性矩阵
    W = S_normalized * S_normalized';  % 余弦相似度矩阵
    % 选择前 k 个特征向量，特征值按升序排序
    [V_S, E_S] = eig(W);
    [sorted_eigenvalues_D, sorted_indices_D] = sort(diag(E_S), 'ascend'); % 改为升序
    X_S = V_S(:, sorted_indices_D(2:k+1));  % 排除第一个特征向量
    
    %% 获得共识嵌入矩阵(CCA)
    XA_A = X_A' * X_A;       
    XS_S = X_S' * X_S;       
    XA_S = X_A' * X_S; 
    XS_A = X_S' * X_A;   
    
    M = [XA_A, XA_S; XS_A, XS_S];
    K = [XA_A, zeros(t,k); zeros(k,t), XS_S]; 
    
    % 广义特征值问题
    [V_P, E_P] = eig(M, K);
    [sorted_eigenvalues_P, sorted_indices_P] = sort(diag(E_P), 'ascend'); % 改为升序
    P = V_P(:, sorted_indices_P(2:l+1));  % 排除第一个特征向量
    
    X = [X_A, X_S] * P;
end
