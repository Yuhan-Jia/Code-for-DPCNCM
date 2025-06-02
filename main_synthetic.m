%% 读取数据

% 设置目标文件夹路径
output_folder = 'result_synthetic_mu'; % 输出文件夹路径
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end
folder_path = 'synthetic_network\mu';  % 数据文件夹
folder_list = dir(folder_path);
subfolders = {folder_list([folder_list.isdir] & ~startsWith({folder_list.name}, '.')).name};
results_table=table();
% 遍历所有文件
for i = 1:length(subfolders)
%for i = 1
    % 获取完整文件路径
    subfolder_path = fullfile(folder_path, subfolders{i});
    name=subfolders{i};
    % 打印当前文件名
    fprintf('正在处理: %s\n', name);

    % 读取 .data 文件
    full_path=fullfile(subfolder_path,'community.dat');
    fid = fopen(full_path, 'r');
    data = textscan(fid, '%d %s', 'Delimiter', '\t', 'MultipleDelimsAsOne', true);
    fclose(fid);

    A=readmatrix(fullfile(subfolder_path,'adjacency_matrix.txt'));
    % 提取节点编号和社区信息
    nodes = data{1};           % 第一列：节点编号
    true_community = data{2}; % 第二列：社区字符串（可能包含多个）

    % 找出所有可能的社区编号
    all_communities = [];
    community_info = cell(length(nodes), 1);

    for i = 1:length(nodes)
        community_list = str2num(true_community{i});  % 将字符串转为数值数组
        community_info{i} = community_list;  % 存储社区编号
        all_communities = [all_communities, community_list]; % 记录所有社区编号
    end

    % 获取社区编号的最大值（确定矩阵列数）
    unique_communities = unique(all_communities);
    num_nodes = max(nodes);  % 确定行数
    num_communities = max(unique_communities); % 确定列数

    % 初始化社区矩阵
    community_matrix = zeros(num_nodes, num_communities);

    % 填充社区矩阵
    for i = 1:length(nodes)
        node = nodes(i);
        communities = community_info{i};
        community_matrix(node, communities) = 1;
    end

    true_community=community_matrix;

    %% 拓扑信息嵌入

    tic;
    % 获取当前 CPU 时间
    startTime = cputime;

    t=5;
    nodes_degree = sum(A, 2); % 节点的度
    % 度矩阵
    D_A = diag(nodes_degree);
    % Laplacian矩阵
    L_A = D_A - A;
    % 选择前 t 个特征向量，特征值按升序排序
    warning('off', 'all');
    [V_A, E_A] = eigs(L_A, D_A, t+1, 'smallestabs');
    [sorted_eigenvalues_A, sorted_indices_A] = sort(diag(E_A), 'ascend'); % 改为升序
    X_A = V_A(:, sorted_indices_A(2:t+1));  % 排除第一个特征向量

    X=X_A;

    %% 确定聚类个数和初始聚类中心
    [D, rho, delta] = ComputeRhoAndDelta(X, 0.02);
    [C_num,initial_centers_id,sorted_id] = FindNumberAndCenter(rho, delta);
    C=size(true_community,2);
    initial_centers_id=sorted_id(1:C);
    initial_centers=X(initial_centers_id,:);

    %% DP-NCM
    % DP_NCM 聚类
    % D: 距离矩阵
    % X: 共识嵌入矩阵
    % C: 社区个数
    % initial_centers: 初始聚类中心
    % epsilon: 终止条件
    % max_iter: 最大迭代次数

    epsilon=0.001;
    max_iter=100;
    w1=0.8;
    w2=0.1;
    w3=0.1;

    %% 初始化参数

    Delta = 0.3;  % 控制离群节点数
    m = 2;
    dc = 0.02;

    [n, ~] = size(X);
    C = size(initial_centers, 1);  % 聚类个数

    T = zeros(n, C);  % 确定隶属度矩阵
    I = zeros(n, 1);  % 不确定隶属度矩阵
    F = zeros(n, 1);  % 离群隶属度矩阵

    % 计算点之间的距离
    D_initial = D(:,initial_centers_id);
    % 处理0距离
    zeroDistIdx = any(D_initial == 0, 2);
    if any(zeroDistIdx)
        [~, zeroCenterIdx] = min(D_initial(zeroDistIdx, :), [], 2);  % 找到最近的聚类中心
        T(zeroDistIdx, :) = 0;  % T的其他列设为0
        T(sub2ind([n, C], find(zeroDistIdx), zeroCenterIdx)) = 1;  % 最近的中心T(i,j) = 1
        I(zeroDistIdx) = 0;  % I(i) = 0
        F(zeroDistIdx) = 0;  % F(i) = 0
    end
    % 处理非0距离
    nonZeroDistIdx = ~zeroDistIdx;
    if any(nonZeroDistIdx)
        T(nonZeroDistIdx, :) = 1 ./ (D_initial(nonZeroDistIdx, :) + 1e-6);

        % 找到最近的两个聚类中心
        [sortedD, sortedIdx] = sort(D_initial(nonZeroDistIdx, :), 2, 'ascend');
        nearestCenter = sortedIdx(:, 1);
        secondNearestCenter = sortedIdx(:, 2);

        % 计算最近的两个聚类中心的中点
        midpoints = (initial_centers(nearestCenter, :) + initial_centers(secondNearestCenter, :)) / 2;

        % 计算与中点的距离
        midpointDistances = sqrt(sum((X(nonZeroDistIdx, :) - midpoints) .^ 2, 2));

        % 初始化T与距离成反比
        I(nonZeroDistIdx) = 1 ./ (midpointDistances + 1e-6);
    end

    % 标准化
    rowSums = sum([T, I], 2);
    T = T ./ rowSums;
    I = I ./ rowSums;

    % 初始化F
    F = 0.5 * ones(n, 1);

    c_centers = initial_centers;

    %%
    % 开始迭代
    for iter = 1:max_iter
        % 更新c_j
        if iter>1
            for j = 1:C
                c_centers(j, :) = sum((w1 .* T(:, j)).^m .* X) / sum((w1 .* T(:, j)).^m);

            end
        end

        % 更新T_{ij}
        for i = 1:n

            D = pdist2(X(i, :), c_centers); % 距离矩阵

            % 判断是否为聚类中心，若是
            [min_distance, closest_center_idx] = min(D);

            if min_distance == 0
                T(i,:)=0;
                T(i,closest_center_idx) = 1;
                I(i) = 0;
                F(i) = 0;
                continue;
            end
            % 若不是
            [~, c_sorted_ind] = sort(D);
            [c_max1, c_max2] = deal(c_sorted_ind(1), c_sorted_ind(2));
            c_max_ba = (c_centers(c_max1, :) + c_centers(c_max2, :)) / 2;

            % 计算K
            K = 1 / (sum(vecnorm(c_centers - X(i, :), 2, 2).^(-2/(m-1))) / w1 + ...
                norm(c_max_ba - X(i, :))^(-2/(m-1)) / w2 + ...
                Delta^(-2/(m-1)) / w3);

            % 更新T_{ij}
            for j = 1:C
                T(i, j) = (K / w1) * norm(X(i, :) - c_centers(j, :))^(-2/(m-1));
            end

            % 更新 I_i, F_i
            [~, c_sorted_ind] = sort(T(i, :));
            [c_max1, c_max2] = deal(c_sorted_ind(1), c_sorted_ind(2));
            c_max_ba = (c_centers(c_max1, :) + c_centers(c_max2, :)) / 2;

            I(i) = (K / w2) * norm(X(i, :) - c_max_ba)^(-2/(m-1));
            F(i) = (K / w3) * Delta^(-2/(m - 1));
        end

        % 收敛性检查
        if iter > 1
            if norm(T - T_prev) < epsilon
                disp('Converged!');
                break;
            end
        end

        % 保存T矩阵作为一步的结果
        T_prev = T;
    end

    % 将每个节点分配到最大的TM值对应的列
    TM = [T,I,F];
    [~, label] = max(TM, [], 2);
    label=int32(label);
    disp('聚类完成.');


    %创建一个记录每个节点所在社区的矩阵

    pre_community = zeros(n, C+2);

    for i = 1:n
        % 找到当前行最大值所在的列
        [maxVal, maxCol] = max(TM(i, :));
        pre_community(i, maxCol) = 1;

        % 如果最大值所在的列是C+1列（重叠社区）
        if maxCol == C+1
            % 找到该行前C列的最大值和次大值
            [~, maxCols] = sort(TM(i, 1:C), 'descend');  % 排序，得到前两大的列的索引
            top2Cols = maxCols(1:2);  % 取前两个最大列

            % 在community矩阵中将这些列的位置赋值为1
            pre_community(i, top2Cols) = 1;
        end

    end

    %% 评价聚类
    pre_community=pre_community(:,1:C);
    % 删除没有真实社区的节点
    zero_indices = find(sum(true_community, 2) == 0);
    A(zero_indices,:)=[];
    A(:,zero_indices)=[];
    pre_community(zero_indices, :) = [];
    true_community(zero_indices, :) = [];

    % 步骤 1：计算每对社区之间的相似度
    [n, label_nums] = size(pre_community);  % 获取节点数和社区数
    unique_trueLabels = 1:label_nums;  % 真实标签中的所有社区
    unique_predLabels = 1:label_nums;  % 预测标签中的所有社区

    % 创建一个相似度矩阵
    similarityMatrix = zeros(label_nums, label_nums);

    % 计算相似度（重叠的节点数量）
    for i = 1:label_nums
        for j = 1:label_nums
            % 找到属于真实社区 i 和预测社区 j 的节点
            trueNodes = find(true_community(:, i) == 1);  % 属于真实社区 i 的节点
            predNodes = find(pre_community(:, j) == 1);  % 属于预测社区 j 的节点

            % 计算这两个社区之间的交集（即共同节点的数量）
            overlap = length(intersect(trueNodes, predNodes));
            similarityMatrix(i, j) = overlap;
        end
    end

    % 步骤 2：根据相似度矩阵重新映射社区标号
    [~, mapping] = max(similarityMatrix, [], 1);  % 对于每个预测社区，找到最相似的真实社区

    % 步骤 3：重新映射 pre_community 中的标号
    newpre_community = zeros(n, label_nums);  % 新的预测社区矩阵
    for i = 1:n
        for j = 1:label_nums
            % 将每个预测社区的标签映射到最相似的真实社区标签
            if pre_community(i, j) == 1
                newpre_community(i, mapping(j)) = 1;  % 映射到真实社区标签
            end
        end
    end


    % 存储聚类结果
    trueNodes_file = fullfile(output_folder, [name '_trueCommunity.csv']);
    preCommunity_file = fullfile(output_folder, [name '_preCommunity.csv']);
    writematrix(true_community, trueNodes_file);
    writematrix(pre_community, preCommunity_file);
    fprintf('结果已保存至: %s\n', output_folder);

end