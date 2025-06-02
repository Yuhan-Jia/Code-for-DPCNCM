function NMI_value = NMI(true_community, pre_community)
%用于计算NMI的函数，输入矩阵

    % 获取节点数
    N = size(true_community, 1);
    
    % 创建混淆矩阵 Z
    unique_true = 1:size(true_community, 2);  % 真实社区的索引
    unique_pred = 1:size(pre_community, 2);  % 预测社区的索引
    
    Z = zeros(length(unique_true), length(unique_pred));  % 初始化混淆矩阵
    
    for i = 1:N
        % 找到每个节点对应的真实社区和预测社区
        true_idx = find(true_community(i, :) == 1);  % 当前节点属于的真实社区
        pred_idx = find(pre_community(i, :) == 1);  % 当前节点属于的预测社区
        Z(true_idx, pred_idx) = Z(true_idx, pred_idx) + 1;
    end
    
    % 归一化混淆矩阵以获得概率
    total_members = sum(Z(:));  % 成员总数
    p_ij = Z / total_members;  % 混淆矩阵的概率分布
    
    % 边际分布
    p_i_dot = sum(p_ij, 2);  % 行和
    p_dot_j = sum(p_ij, 1);  % 列和
    
    % 防止 log(0)，将零概率替换为一个小值
    p_ij(p_ij == 0) = eps;
    p_i_dot(p_i_dot == 0) = eps;
    p_dot_j(p_dot_j == 0) = eps;
    
    % 计算 NMI 的分子
    term1 = sum(sum(p_ij .* log(p_ij ./ (p_i_dot * p_dot_j))));
    
    % 计算 NMI 的分母
    term2 = sum(p_i_dot .* log(p_i_dot)) + sum(p_dot_j .* log(p_dot_j));
    
    % 计算 NMI
    NMI_value = -2 * term1 / term2;
end