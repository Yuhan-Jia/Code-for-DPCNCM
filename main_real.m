
% DP_NCM 聚类，没有加入参数更新
% D: 距离矩阵
% X: 共识嵌入矩阵
% C: 社区个数
% initial_centers: 初始聚类中心
% epsilon: 终止条件
% max_iter: 最大迭代次数

% 读取网络数据
name=char("Wisconsin");
load("realworld_network/Wisconsin.mat");

A=adj_matrix;
S=value_matrix;

% 设置目标文件夹路径
output_folder = 'result_real'; % 替换为你的输出文件夹路径
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% 获得共识矩阵

% 此处有标签的真实网络(football,polbooks)与其他网络的处理不同，计算的指标也不同
if strcmp(name, 'football')
    % 拓扑信息嵌入
    t=5;
    nodes_degree = sum(A, 2); % 节点的度
    % 度矩阵
    D_A = diag(nodes_degree);
    % Laplacian矩阵
    L_A = D_A - A;
    % 选择前 t 个特征向量，特征值按升序排序
    [V_A, E_A] = eig(L_A, D_A);
    [sorted_eigenvalues_A, sorted_indices_A] = sort(diag(E_A), 'ascend'); % 改为升序
    X_A = V_A(:, sorted_indices_A(2:t+1));  % 排除第一个特征向量
    X=X_A;
    [D, rho, delta] = ComputeRhoAndDelta(X,0.1);
    D = pdist2(X, X);
    [~,~,sorted_id] = FindNumberAndCenter(rho, delta);
    C=size(S,2);
    initial_centers_id=sorted_id(1:C);
    initial_centers=X(initial_centers_id,:);
end


if strcmp(name, 'polbooks')
    % 拓扑信息嵌入
    t=2;
    nodes_degree = sum(A, 2); % 节点的度
    % 度矩阵
    D_A = diag(nodes_degree);
    % Laplacian矩阵
    L_A = D_A - A;
    % 选择前 t 个特征向量，特征值按升序排序
    [V_A, E_A] = eig(L_A, D_A);
    [sorted_eigenvalues_A, sorted_indices_A] = sort(diag(E_A), 'ascend'); % 改为升序
    X_A = V_A(:, sorted_indices_A(2:t+1));  % 排除第一个特征向量
    X=X_A;
    [D, rho, delta] = ComputeRhoAndDelta(X,0.1);
    D = pdist2(X, X);
    [~,~,sorted_id] = FindNumberAndCenter(rho, delta);
    C=size(S,2);
    initial_centers_id=sorted_id(1:C);
    initial_centers=X(initial_centers_id,:);
end

if strcmp(name, 'Cornell')
    t=6;k=4;l=8;dc=0.02;
    X=getEmbeddingMatrix(A,S,t,k,l);
    [D, rho, delta] = ComputeRhoAndDelta(X,dc);
    [C,initial_centers_id,sorted_id] = FindNumberAndCenter(rho, delta);
    initial_centers_id=sorted_id(1:C);
    initial_centers=X(initial_centers_id,:);
end
if strcmp(name, 'Texas')
    t=7;k=7;l=7;dc=0.02;
    X=getEmbeddingMatrix(A,S,t,k,l);
    [D, rho, delta] = ComputeRhoAndDelta(X,dc);
    [C,initial_centers_id,sorted_id] = FindNumberAndCenter(rho, delta);
    initial_centers_id=sorted_id(1:C);
    initial_centers=X(initial_centers_id,:);
end
if strcmp(name, 'Wisconsin')
    t=7;k=7;l=7;dc=0.2;
    X=getEmbeddingMatrix(A,S,t,k,l);
    [D, rho, delta] = ComputeRhoAndDelta(X,dc);
    [C,initial_centers_id,sorted_id] = FindNumberAndCenter(rho, delta);
    initial_centers_id=sorted_id(1:C);
    initial_centers=X(initial_centers_id,:);
end



%% DP-NCM

epsilon=0.001;
max_iter=100;
w1=0.33;
w2=0.33;
w3=0.33;



Delta = 1;
m = 2;

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
disp('聚类完成.')

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
        top1Cols = maxCols(1);  % 取前1个最大列

        % 在community矩阵中将这些列的位置赋值为1
        pre_community(i, top1Cols) = 1;

    end

end

overlap_index=find(pre_community(:,C+1)~=0);

%% 评价聚类
pre_community=pre_community(:,1:C);

% 若有Label
if strcmp(name, 'football')||strcmp(name, 'polbooks')
    true_community=S;
    nmi_val = NMI(S, pre_community);
    fprintf('NMI: %.4f\n', nmi_val);
end

% 若无Label
if strcmp(name, 'Cornell')||strcmp(name, 'Texas')||strcmp(name, 'Wisconsin')
    % 计算模块度
    EQ_val = EQ(A, pre_community);
    fprintf('EQ_val: %.4f\n', EQ_val);
end


