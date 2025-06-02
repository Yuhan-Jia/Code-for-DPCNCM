function EQ_value = EQ(A, community_matrix)
% ExtendedModularity 计算扩展模块度 EQ，用于重叠社区划分
%
% 输入:
%   A               : 邻接矩阵 (n x n)，假设为无向图
%   community_matrix: 社区隶属矩阵 (n x K)，如果节点 i 属于社区 k，则 
%                     community_matrix(i, k) = 1，否则为 0
%
% 输出:
%   EQ              : 扩展模块度指标
%
% 公式:
%   EQ = (1/(2m)) * sum_{k=1}^{K} sum_{i,j in C_k} (1/(O_i * O_j)) * [ A(i,j) - (k_i*k_j)/(2m) ]
%
%   其中:
%     m   : 图中边的数量，即 m = sum(A(:))/2
%     k_i : 节点 i 的度数, k_i = sum(A(i,:))
%     O_i : 节点 i 的重叠度, O_i = sum(community_matrix(i,:))

    % 获取节点数和社区数
    n = size(A, 1);
    K = size(community_matrix, 2);
    
    % 计算图中边数
    m = sum(A(:)) / 2;
    
    % 计算每个节点的度数
    degrees = sum(A, 2);
    
    % 计算每个节点的重叠度 O_i
    O = sum(community_matrix, 2);
    
    % 初始化扩展模块度求和变量
    EQ_sum = 0;
    
    % 对每个社区进行累加
    for k = 1:K
        % 找出属于第 k 个社区的节点索引
        nodes_in_comm = find(community_matrix(:, k) > 0);
        
        % 对社区内所有节点对 (i, j) 求和
        for ii = 1:length(nodes_in_comm)
            i = nodes_in_comm(ii);
            for jj = 1:length(nodes_in_comm)
                j = nodes_in_comm(jj);
                % 权重为 1/(O_i * O_j)
                weight = 1 / (O(i) * O(j));
                % 节点对 (i,j) 的贡献
                contribution = weight * (A(i,j) - (degrees(i) * degrees(j)) / (2*m));
                EQ_sum = EQ_sum + contribution;
            end
        end
    end
    
    % 最终 EQ 值
    EQ_value = EQ_sum / (2*m);
end
