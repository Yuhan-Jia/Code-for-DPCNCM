function [c,initial_centers_id,sorted_indices]=FindNumberAndCenter(rho, delta)
    % 找到初始聚类中心和最佳聚类个数的函数
    % rho: 所有节点的局部密度
    % delta: 所有节点的最短距离
    
    % 计算CS值
    CS = rho .* delta;
    [sorted_CS, sorted_indices] = sort(CS, 'descend'); % 按照CS值降序排序
    x = (1:length(CS))';  % 横坐标从 1 到 n
    
    width_cm =5;
    height_cm = 4;
     % 绘制CS值的折线图
    %figure;
    % 创建绘图
    % set(gcf, 'PaperUnits', 'centimeters'); % 使用厘米为单位
    % set(gcf, 'PaperSize', [width_cm, height_cm]); % 设置 PDF 页面尺寸
    % set(gcf, 'PaperPosition', [0, 0, width_cm, height_cm]); % 设置图在页面上的位置和大小
    % set(gcf, 'PaperPositionMode', 'manual'); % 自动填充为全框
    % plot(x, sorted_CS, '-o', 'LineWidth', 1, 'MarkerSize', 2, 'MarkerFaceColor', 'k');
    % %title('Sorted CS Vector', 'FontSize', 14);
    % xlabel('Index', 'FontSize', 8);
    % ylabel('CS Value', 'FontSize', 8);
    % hold on;
    % 
    % % 寻找“膝点”（拐点），并拟合两条直线
     % 假设找到最佳聚类数时的拐点位置为c
    c = FindOptimalK(x, sorted_CS);  % 计算最佳的k值，即聚类数
    initial_centers_id=sorted_indices(1:c);


    % 拟合左侧和右侧的线性模型
    [a0_left, a1_left] = FitLinearModel(x(1:c), sorted_CS(1:c)); % 左侧线性拟合
    [a0_right, a1_right] = FitLinearModel(x(c+1:end), sorted_CS(c+1:end)); % 右侧线性拟合
    % 
    % % 画出拟合的直线
    % plot(x(1:c), a0_left + a1_left * x(1:c), 'r', 'LineWidth', 1); % 左侧拟合线
    % plot(x(c+1:end), a0_right + a1_right * x(c+1:end), 'r', 'LineWidth',1); % 右侧拟合线
    % ylim([-0.5, max(sorted_CS) + 0.1]); % y 轴最小值设为 -0.5，最大值略大于最大 CS 值
    % set(gca,  'LineWidth', 0.8, 'TickDir', 'in','Box', 'on','FontName', 'Times New Roman','FontWeight','normal');

    
    %hold off;
end

% 寻找最佳聚类数k，返回膝点位置
function c = FindOptimalK(x, sorted_CS)
    min_rmse = Inf;  % 初始最小RMSE
    c = 2;  % 从2开始测试

    for k = 2:min(30,size(x,1))
        % 拟合左侧和右侧的线性模型
        [a0_left, a1_left] = FitLinearModel(x(1:k), sorted_CS(1:k));  % 左侧拟合
        [a0_right, a1_right] = FitLinearModel(x(k+1:end), sorted_CS(k+1:end));  % 右侧拟合
        
        % 计算RMSE
        rmse_left = RMSE(x(1:k), sorted_CS(1:k), a0_left, a1_left);
        rmse_right = RMSE(x(k+1:end), sorted_CS(k+1:end), a0_right, a1_right);
        
        % 计算整体RMSE
        rmse_c = (k-1)/(length(x)-1) * rmse_left + (length(x)-k)/(length(x)-1) * rmse_right;
        
        % 找到最小RMSE对应的k
        if rmse_c < min_rmse
            min_rmse = rmse_c;
            c = k;
        end
    end
end

% 线性拟合函数：拟合数据点，返回参数a0, a1
function [a0, a1] = FitLinearModel(x, y)
    n = length(x);
    a1 = (n * sum(x .* y) - sum(x) * sum(y)) / (n * sum(x.^2) - (sum(x))^2);
    a0 = (sum(y) - a1 * sum(x)) / n;
end

% 计算RMSE
function rmse = RMSE(x, y, a0, a1)
    predicted_y = a0 + a1 * x;
    rmse = sqrt(sum((y - predicted_y).^2) / length(x));
end


