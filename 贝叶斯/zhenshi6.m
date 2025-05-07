clc;
clear;
close all;

% 全局变量
global N delta_t T k w_k S_yn;

% 读取真实数据
file_path = 'data2025.mat';  % 修改为你的数据文件路径

if ~exist(file_path, 'file')
    error('找不到数据文件: %s', file_path);
end

% 加载数据
data = load(file_path);

% 获取测量数据
if isfield(data, 'measurement')
    z = data.measurement;
elseif isfield(data, 'z')
    z = data.z;
else
    % 如果不确定变量名，可以查看数据结构
    disp('数据文件中的变量：');
    fieldnames(data)
    error('请指定正确的测量数据变量名');
end

% 获取参数
if isfield(data, 'T')
    T = data.T;
else
    T = 600;
    warning('未找到T，使用默认值 %d', T);
end

if isfield(data, 'delta_t')
    delta_t = data.delta_t;
else
    delta_t = 0.005;
    warning('未找到delta_t，使用默认值 %.3f', delta_t);
end

% 选择数据时间段
time_start = 0;  % 开始时间（秒）
time_end = 600;  % 结束时间（秒）- 可以调整此值以查看不同时段的频谱特征

% 计算索引范围
start_idx = max(1, round(time_start/delta_t));
end_idx = min(length(z), round(time_end/delta_t));

% 提取时间段数据
z_segment = z(start_idx:end_idx);
T_segment = (end_idx - start_idx + 1) * delta_t;  % 重新计算总时间

% 更新数据长度
N = length(z_segment);

% 输出数据信息
disp('数据信息:');
disp(['总数据点数: ' num2str(length(z))]);
disp(['分析数据点数: ' num2str(N)]);
disp(['总时间: ' num2str(T) ' s']);
disp(['分析时间段: ' num2str(time_start) ' s 到 ' num2str(time_end) ' s']);
disp(['时间步长: ' num2str(delta_t) ' s']);
disp(['采样频率: ' num2str(1/delta_t) ' Hz']);

% 开始真实数据分析
disp(' ');
disp('开始真实数据分析...');

%% 改进的频谱计算与峰值检测
% ==============================================

% Step 1: 计算原始频谱
k = 1500;  % 增加频谱点数以覆盖更大频率范围
w_k = (2 * pi * (1:k)) / T_segment;
part1 = zeros(1, k);
for n = 1:N-1
    part1 = part1 + z_segment(n+1) * exp(-1i * w_k * n * delta_t);
end
part1 = part1 + z_segment(1);
S_yn = (delta_t / (2 * pi * N)) * abs(part1).^2;

% Step 2: 多种平滑方法对比
% 2.1 移动平均平滑 - 不同窗口大小
window_sizes = [5, 9, 15, 21];
S_yn_movmean = cell(length(window_sizes), 1);

for i = 1:length(window_sizes)
    % 自定义移动平均函数（不依赖工具箱）
    window_size = window_sizes(i);
    padded_S_yn = [ones(1,floor(window_size/2))*S_yn(1), S_yn, ones(1,floor(window_size/2))*S_yn(end)];
    smoothed = zeros(size(S_yn));
    
    for j = 1:length(S_yn)
        smoothed(j) = mean(padded_S_yn(j:j+window_size-1));
    end
    
    S_yn_movmean{i} = smoothed;
end

% 2.2 自定义设计的多项式平滑（替代Savitzky-Golay）
% 实现一个简单的多项式平滑，使用局部多项式拟合
window_size = 21;
half_window = floor(window_size/2);
S_yn_poly_smooth = zeros(size(S_yn));

for i = 1:length(S_yn)
    % 确定局部窗口的范围
    start_idx = max(1, i - half_window);
    end_idx = min(length(S_yn), i + half_window);
    
    % 构建局部坐标系
    x_local = (start_idx:end_idx)' - i;
    y_local = S_yn(start_idx:end_idx)';
    
    % 3阶多项式拟合
    if length(x_local) > 3  % 确保有足够的点进行拟合
        p = polyfit(x_local, log10(y_local), 3);  % 对对数数据进行拟合
        S_yn_poly_smooth(i) = 10^polyval(p, 0);  % 计算中心点的平滑值
    else
        S_yn_poly_smooth(i) = S_yn(i);  % 如果点不够，保持原值
    end
end

% 2.3 使用加权平均作为高斯平滑的替代
% 创建类高斯权重
weights = exp(-(-10:10).^2/(2*5^2));
weights = weights / sum(weights);  % 归一化
S_yn_weighted = zeros(size(S_yn));

% 填充数据以处理边界
padded_data = [ones(1,10)*S_yn(1), S_yn, ones(1,10)*S_yn(end)];

for i = 1:length(S_yn)
    S_yn_weighted(i) = sum(padded_data(i:i+20) .* weights);
end

% Step 3: 准备比较不同平滑方法的结果
figure('Position', [100, 500, 1200, 700]);

% 3.1 原始频谱
subplot(2, 3, 1);
semilogy(w_k, S_yn, 'b-', 'LineWidth', 1);
title('原始频谱', 'FontSize', 12);
xlabel('\omega_k (rad/s)', 'FontSize', 10);
ylabel('S_{yn}(\omega_k)', 'FontSize', 10);
grid on;
xlim([0, 10]);

% 3.2 不同窗口大小的移动平均
subplot(2, 3, 2);
colors = {'r-', 'g-', 'c-', 'm-'};
hold on;
for i = 1:length(window_sizes)
    semilogy(w_k, S_yn_movmean{i}, colors{i}, 'LineWidth', 1);
end
title('移动平均平滑', 'FontSize', 12);
xlabel('\omega_k (rad/s)', 'FontSize', 10);
ylabel('S_{yn}(\omega_k)', 'FontSize', 10);
grid on;
xlim([0, 10]);
legend(arrayfun(@(x) sprintf('窗口大小=%d', x), window_sizes, 'UniformOutput', false), 'Location', 'northeast');

% 3.3 多项式平滑
subplot(2, 3, 3);
semilogy(w_k, S_yn_poly_smooth, 'm-', 'LineWidth', 1);
title('多项式平滑', 'FontSize', 12);
xlabel('\omega_k (rad/s)', 'FontSize', 10);
ylabel('S_{yn}(\omega_k)', 'FontSize', 10);
grid on;
xlim([0, 10]);

% 3.4 加权平均平滑
subplot(2, 3, 4);
semilogy(w_k, S_yn_weighted, 'k-', 'LineWidth', 1);
title('加权平均平滑', 'FontSize', 12);
xlabel('\omega_k (rad/s)', 'FontSize', 10);
ylabel('S_{yn}(\omega_k)', 'FontSize', 10);
grid on;
xlim([0, 10]);

% Step 4: 高级峰值检测

% 4.1 设置最优平滑方法 - 使用多项式平滑
S_best = S_yn_poly_smooth;

% 4.2 频率范围限制
min_freq = 0.1;  % 最小频率 (rad/s)
max_freq = 10;   % 最大频率 (rad/s)
valid_indices = find(w_k >= min_freq & w_k <= max_freq);

% 4.3 寻找所有峰值 - 自定义findpeaks函数
% 定义一个简单的峰值检测函数
log_S = log10(S_best);
peaks = [];
locs = [];
widths = [];
prominences = [];

% 最小峰间距（索引）
min_peak_distance = round(0.1 / (w_k(2) - w_k(1)));
% 最小峰值突显度
min_peak_prominence = 0.5;

for i = 2:length(valid_indices)-1
    idx = valid_indices(i);
    if log_S(idx) > log_S(idx-1) && log_S(idx) > log_S(idx+1)
        % 找到一个局部极大值
        
        % 计算左侧最近的峰谷
        left_idx = idx;
        while left_idx > 1 && log_S(left_idx) >= log_S(left_idx-1)
            left_idx = left_idx - 1;
        end
        
        % 计算右侧最近的峰谷
        right_idx = idx;
        while right_idx < length(log_S) && log_S(right_idx) >= log_S(right_idx+1)
            right_idx = right_idx + 1;
        end
        
        % 计算突显度（prominence）
        left_min = min(log_S(left_idx:idx));
        right_min = min(log_S(idx:right_idx));
        ref_min = max(left_min, right_min);
        prominence = log_S(idx) - ref_min;
        
        % 如果突显度足够大，记录这个峰
        if prominence >= min_peak_prominence
            % 确保与之前找到的峰距离足够远
            if isempty(locs) || min(abs(idx - locs)) > min_peak_distance
                peaks = [peaks, log_S(idx)];
                locs = [locs, idx];
                
                % 计算半高宽（width）
                half_height = log_S(idx) - prominence/2;
                
                % 找到半高点的左侧和右侧索引
                left_half = idx;
                while left_half > 1 && log_S(left_half) >= half_height
                    left_half = left_half - 1;
                end
                
                right_half = idx;
                while right_half < length(log_S) && log_S(right_half) >= half_height
                    right_half = right_half + 1;
                end
                
                width = w_k(right_half) - w_k(left_half);
                widths = [widths, width];
                prominences = [prominences, prominence];
            end
        end
    end
end

% 从索引转换回频率
locs_freq = w_k(locs);

% 4.4 按照突显度（prominence）排序峰值
[sorted_prominences, sort_idx] = sort(prominences, 'descend');
sorted_peaks = peaks(sort_idx);
sorted_locs = locs_freq(sort_idx);
sorted_widths = widths(sort_idx);

% 4.5 提取前N个最显著的峰
num_peaks_to_show = min(5, length(sorted_peaks));
if num_peaks_to_show > 0
    top_peaks = 10.^sorted_peaks(1:num_peaks_to_show);
    top_locs = sorted_locs(1:num_peaks_to_show);
    top_prominences = sorted_prominences(1:num_peaks_to_show);
    top_widths = sorted_widths(1:num_peaks_to_show);

    % 4.6 亚像素峰值定位优化 (对每个峰值进行抛物线拟合)
    precise_locs = zeros(size(top_locs));
    precise_peaks = zeros(size(top_peaks));

    for i = 1:length(top_locs)
        % 找到最接近的索引
        [~, idx] = min(abs(w_k - top_locs(i)));
        
        % 确保有足够的点进行拟合
        if idx > 2 && idx < length(log_S)-2
            % 取5个点进行拟合
            x = w_k(idx-2:idx+2);
            y = log_S(idx-2:idx+2);
            
            % 抛物线拟合
            p = polyfit(x, y, 2);
            
            % 通过导数找到精确峰值位置
            precise_locs(i) = -p(2)/(2*p(1));
            
            % 计算对应的峰值
            precise_peaks(i) = 10^polyval(p, precise_locs(i));
        else
            % 如果无法拟合，使用原始值
            precise_locs(i) = top_locs(i);
            precise_peaks(i) = top_peaks(i);
        end
    end

    % Step 5: 绘制最终检测结果
    subplot(2, 3, [5,6]);
    semilogy(w_k, S_best, 'b-', 'LineWidth', 1.5);
    hold on;

    % 标注所有找到的主要峰值
    colors = {'r', 'g', 'm', 'c', 'y'};
    for i = 1:num_peaks_to_show
        semilogy(precise_locs(i), precise_peaks(i), [colors{mod(i-1,length(colors))+1}, 'o'], ...
            'MarkerSize', 8+2*(num_peaks_to_show-i), 'LineWidth', 2);
        
        % 为每个峰添加文本标注
        text_x = precise_locs(i) + 0.05;
        text_y = precise_peaks(i) * (1.2 + 0.2*i);
        
        % 突显度可以理解为峰的"重要性"
        text(text_x, text_y, sprintf('峰值 #%d: %.3f rad/s\n(%.3f Hz)\n突显度: %.2f', ...
            i, precise_locs(i), precise_locs(i)/(2*pi), top_prominences(i)), ...
            'FontSize', 9, 'Color', colors{mod(i-1,length(colors))+1}, 'FontWeight', 'bold');
    end

    % 添加图例和标签
    title('高级峰值检测结果', 'FontSize', 14);
    xlabel('\omega_k (rad/s)', 'FontSize', 12);
    ylabel('S_{yn}(\omega_k)', 'FontSize', 12);
    grid on;
    xlim([0, 10]);

    % 添加频率域搜索范围标示
    x_range = [min_freq, max_freq];
    y_range = get(gca, 'YLim');
    patch([x_range(1) x_range(1) x_range(2) x_range(2)], [y_range(1) y_range(2) y_range(2) y_range(1)], ...
        [0.9 0.9 1], 'FaceAlpha', 0.1, 'EdgeColor', 'none');

    % 显示主峰的详细信息
    main_peak_freq = precise_locs(1);
    main_peak_hz = main_peak_freq/(2*pi);

    % 输出主峰信息
    fprintf('---------- 主峰信息 ----------\n');
    fprintf('频率: %.4f rad/s (%.4f Hz)\n', main_peak_freq, main_peak_hz);
    fprintf('峰值幅度: %.4e\n', precise_peaks(1));
    fprintf('相对突显度: %.4f\n', top_prominences(1));
    fprintf('半高宽: %.4f rad/s\n', top_widths(1));
    fprintf('\n');

    % 输出所有检测到的峰值
    fprintf('---------- 检测到的所有主要峰值 ----------\n');
    fprintf('序号  频率(rad/s)  频率(Hz)    突显度    半高宽(rad/s)\n');
    for i = 1:num_peaks_to_show
        fprintf('#%-2d   %-10.4f  %-10.4f  %-10.4f  %-10.4f\n', ...
            i, precise_locs(i), precise_locs(i)/(2*pi), top_prominences(i), top_widths(i));
    end

    % 添加总结性文本框到图中
    annotation('textbox', [0.02, 0.02, 0.44, 0.15], 'String', ...
        sprintf(['高级峰值检测结果:\n' ...
                 '主峰: %.4f rad/s (%.4f Hz)\n' ...
                 '使用方法: 多项式平滑 + 自定义峰值检测 + 抛物线插值\n' ...
                 '分析区间: %.1f-%.1f rad/s\n' ...
                 '检测到 %d 个显著峰值'], ...
                 main_peak_freq, main_peak_hz, min_freq, max_freq, num_peaks_to_show), ...
        'FontSize', 10, 'BackgroundColor', [0.95, 0.95, 0.95], 'EdgeColor', 'black');

    %% 使用最准确的峰值进行后续参数估计
    % 使用高级检测方法找到的更精确的主峰作为初始值
    estimated_omega = main_peak_freq;

    % PPT第7页: 参数估计
    % 调整初始值以确保主峰附近的参数
    x0 = [estimated_omega, 0.05, precise_peaks(1), 0.01];
    disp(['初始值: [' num2str(x0) ']']);

else
    % 如果没有找到有效峰值
    warning('未检测到显著的峰值。');
    % 使用原始方法找到的最大值作为备选
    [~, max_idx] = max(S_yn(valid_indices));
    estimated_omega = w_k(valid_indices(max_idx));
    x0 = [estimated_omega, 0.05, S_yn(valid_indices(max_idx)), 0.01];
    disp(['未找到显著峰值，使用最大值作为初始值: [' num2str(x0) ']']);
end

% PPT第19页(Function部分): 目标函数定义
fun = @fun_BSDA;

% 使用多步优化方法
x_current = x0;

% 第一步：网格搜索
disp(' ');
disp('第一步：网格搜索优化...');
for i = 1:4
    x_temp = x_current;
    
    if i == 1  % omega
        search_range = (x_current(1) * 0.8):(x_current(1) * 0.05):(x_current(1) * 1.2);
    elseif i == 2  % zeta
        search_range = 0.001:0.005:0.2;
    elseif i == 3  % S_f0
        search_range = (x_current(3) * 0.1):(x_current(3) * 0.2):(x_current(3) * 2);
    else  % sigma_e^2
        search_range = 0.0001:0.005:0.5;
    end
    
    best_val = Inf;
    best_param = x_current(i);
    
    for val = search_range
        x_temp(i) = val;
        f_val = fun(x_temp);
        if f_val < best_val
            best_val = f_val;
            best_param = val;
        end
    end
    
    x_current(i) = best_param;
end

disp('第一步结果:');
disp(['omega: ' num2str(x_current(1))]);
disp(['zeta: ' num2str(x_current(2))]);
disp(['S_f0: ' num2str(x_current(3))]);
disp(['sigma_e^2: ' num2str(x_current(4))]);
disp(['函数值: ' num2str(fun(x_current))]);

% 第二步：精细优化
disp(' ');
disp('第二步：精细优化...');
options = optimset('MaxFunEvals', 20000, 'MaxIter', 10000, 'TolFun', 1e-8, 'TolX', 1e-8);
[x_opt, fval, exitflag, output] = fminsearch(fun, x_current, options);

% 输出优化结果
disp('优化结果:');
disp(['自然频率 (ω): ' num2str(x_opt(1)) ' rad/s']);
disp(['频率 (f): ' num2str(x_opt(1)/(2*pi)) ' Hz']);
disp(['阻尼比 (ζ): ' num2str(x_opt(2))]);
disp(['白噪声谱强度 (S_f0): ' num2str(x_opt(3))]);
disp(['测量噪声方差 (σ_e^2): ' num2str(x_opt(4))]);

% PPT第7页: 不确定性量化
theta_n = 4;
H = zeros(theta_n, theta_n);
dtheta = 0.0001;

% 确保σ_e^2不会太小
x_opt(4) = max(x_opt(4), 0.001);

for n = 1:theta_n
    dTheta_i = zeros(1, theta_n);
    dTheta_i(n) = dtheta;
    
    y1 = x_opt + dTheta_i;
    y2 = x_opt - dTheta_i;
    
    % 确保参数边界
    if n == 4 % σ_e^2
        y1(4) = max(y1(4), 0.001);
        y2(4) = max(y2(4), 0.001);
    end
    
    H(n,n) = (fun_BSDA(y1) - 2.0 * fun_BSDA(x_opt) + fun_BSDA(y2)) / (dtheta * dtheta);
end

for n = 1:theta_n
    dTheta_i = zeros(1, theta_n);
    dTheta_i(n) = dtheta;
    for m = 1:theta_n
        if n == m
            continue;
        end
        dTheta_j = zeros(1, theta_n);
        dTheta_j(m) = dtheta;
        
        temp1 = x_opt + dTheta_i + dTheta_j;
        temp2 = x_opt + dTheta_i - dTheta_j;
        temp3 = x_opt - dTheta_i + dTheta_j;
        temp4 = x_opt - dTheta_i - dTheta_j;
        
        % 确保参数边界
        if n == 4
            temp1(4) = max(temp1(4), 0.001);
            temp2(4) = max(temp2(4), 0.001);
            temp3(4) = max(temp3(4), 0.001);
            temp4(4) = max(temp4(4), 0.001);
        end
        if m == 4
            temp1(4) = max(temp1(4), 0.001);
            temp2(4) = max(temp2(4), 0.001);
            temp3(4) = max(temp3(4), 0.001);
            temp4(4) = max(temp4(4), 0.001);
        end
        
        H(n,m) = (fun_BSDA(temp1) - fun_BSDA(temp2) - fun_BSDA(temp3) + fun_BSDA(temp4)) / (4.0 * dtheta * dtheta);
    end
end

% 检查Hessian矩阵是否可逆
if rcond(H) < eps || any(isnan(H(:))) || any(isinf(H(:)))
    disp('Warning: Hessian matrix is ill-conditioned. Using pseudo-inverse.');
    sigma = pinv(H + 1e-10 * eye(size(H)));  % 添加正则化项
else
    sigma = inv(H);
end

% 计算标准差和变异系数
SD = sqrt(abs(diag(sigma)));  % 取绝对值以避免虚数
COV = SD ./ abs(x_opt(:));

% 正态分布PDF函数（替代normpdf，不需要统计工具箱）
normal_pdf = @(x, mu, sigma) (1./(sigma * sqrt(2*pi))) .* exp(-0.5 * ((x - mu)./sigma).^2);

% PPT第9页: 高斯分布图（左上 - omega，右上 - zeta）
figure('Position', [100, 600, 1000, 400]);

% 第1个图：自然频率(ω)的高斯分布（左）
subplot(1,2,1);
mu1 = x_opt(1);
sigmaPDF1 = SD(1);
xxx = linspace(mu1 - 4*sigmaPDF1, mu1 + 4*sigmaPDF1, 1000);
yyy = normal_pdf(xxx, mu1, sigmaPDF1);
plot(xxx, yyy, 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410]);
grid on;
title('自然频率(ω)高斯分布');
xlabel('频率 (rad/s)');
ylabel('概率密度');

% 第2个图：阻尼比(ζ)的高斯分布（右）
subplot(1,2,2);
mu2 = x_opt(2);
sigmaPDF2 = SD(2);
xxx = linspace(mu2 - 4*sigmaPDF2, mu2 + 4*sigmaPDF2, 1000);
yyy = normal_pdf(xxx, mu2, sigmaPDF2);
plot(xxx, yyy, 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410]);
grid on;
title('阻尼比(ζ)高斯分布');
xlabel('阻尼比');
ylabel('概率密度');

% PPT第10页: 高斯分布图（左 - S_f0，右 - sigma_e^2）
figure('Position', [100, 200, 1000, 400]);

% 第3个图：白噪声谱强度(S_f0)的高斯分布（左）
subplot(1,2,1);
mu3 = x_opt(3);
sigmaPDF3 = SD(3);
xxx = linspace(mu3 - 4*sigmaPDF3, mu3 + 4*sigmaPDF3, 1000);
yyy = normal_pdf(xxx, mu3, sigmaPDF3);
plot(xxx, yyy, 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410]);
grid on;
title('白噪声谱强度(S_{f0})高斯分布');
xlabel('S_{f0}');
ylabel('概率密度');

% 第4个图：测量噪声方差(σ_e^2)的高斯分布（右）
subplot(1,2,2);
mu4 = x_opt(4);
sigmaPDF4 = SD(4);
xxx = linspace(max(0, mu4 - 4*sigmaPDF4), mu4 + 4*sigmaPDF4, 1000);
yyy = normal_pdf(xxx, mu4, sigmaPDF4);
plot(xxx, yyy, 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410]);
grid on;
title('测量噪声方差(σ_e^2)高斯分布');
xlabel('σ_e^2');
ylabel('概率密度');

% 输出不确定性结果
disp(' ');
disp('不确定性量化结果:');
disp('参数         | 优化值     | 标准差(SD) | 变异系数(COV)');
disp('--------------------------------------------------------');

% 直接格式化输出
row1 = sprintf('自然频率 (ω)    | %.4f | %.4f | %.4f', x_opt(1), SD(1), COV(1));
row2 = sprintf('阻尼比 (ζ)      | %.4f | %.4f | %.4f', x_opt(2), SD(2), COV(2));
row3 = sprintf('白噪声谱强度 (S_f0) | %.4f | %.4f | %.4f', x_opt(3), SD(3), COV(3));
row4 = sprintf('测量噪声方差 (σ_e^2) | %.4f | %.4f | %.4f', x_opt(4), SD(4), COV(4));

disp(row1);
disp(row2);
disp(row3);
disp(row4);

% PPT第12页: 测量频谱与理论频谱对比图
figure('Position', [100, 50, 800, 500]);

% 绘制测量频谱
semilogy(w_k, S_best, 'b-', 'LineWidth', 1.5);
hold on;

% 计算理论频谱曲线
omiga_opt = x_opt(1);
zeta_opt = x_opt(2);
S_f0_opt = x_opt(3);
sigma_e_opt = x_opt(4);

% 计算理论频谱
R = zeros(1, N);
for n = 1:N-1
    tau = n * delta_t;
    if zeta_opt == 0
        R(n) = cos(omiga_opt * tau) * pi * S_f0_opt / (2.0 * omiga_opt^3);
    else
        Omiga_d = sqrt(1 - zeta_opt^2) * omiga_opt;
        R(n) = (cos(Omiga_d * tau) + sin(Omiga_d * tau) * zeta_opt / sqrt(1 - zeta_opt^2)) * ...
               exp(-zeta_opt * omiga_opt * tau) * pi * S_f0_opt / (2.0 * omiga_opt^3 * zeta_opt);
    end
end

temp3 = (sigma_e_opt^2 * delta_t) / (2 * pi);
temp4 = delta_t / (2 * pi * N);
En = real(fft([N, 2*(N-1:-1:2).*R(2:N-1)]));
E_theoretical = temp4 * En + temp3;

semilogy(w_k, E_theoretical(2:k+1), 'r--', 'LineWidth', 1.5);
grid on;
xlabel('w_k');
ylabel('S_{yn}(w_k)');
ylim([min(S_best(S_best > 0)) * 0.1, max(S_best) * 10]);
xlim([0, 15]);
title('改进后的测量与理论频谱对比');
legend('测量频谱', '理论频谱', 'Location', 'northeast');

% 添加主峰标记
if exist('main_peak_freq', 'var')
    hold on;
    semilogy(main_peak_freq, precise_peaks(1), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    text(main_peak_freq+0.1, precise_peaks(1)*1.5, sprintf('主峰: %.3f rad/s (%.3f Hz)', main_peak_freq, main_peak_hz), ...
         'FontSize', 12, 'Color', 'red', 'FontWeight', 'bold');

    % 添加信息标注
    annotation('textbox', [0.02, 0.02, 0.35, 0.15], 'String', ...
        sprintf(['使用高级峰值检测:\n' ...
                 '主峰频率: %.4f rad/s (%.4f Hz)\n' ...
                 '优化后频率: %.4f rad/s (%.4f Hz)\n' ...
                 '阻尼比: %.4f'], ...
                 main_peak_freq, main_peak_hz, x_opt(1), x_opt(1)/(2*pi), x_opt(2)), ...
        'FontSize', 10, 'BackgroundColor', [0.95, 0.95, 0.95], 'EdgeColor', 'black');
else
    % 如果没有找到主峰，直接显示优化结果
    annotation('textbox', [0.02, 0.02, 0.35, 0.15], 'String', ...
        sprintf(['优化结果:\n' ...
                 '频率: %.4f rad/s (%.4f Hz)\n' ...
                 '阻尼比: %.4f'], ...
                 x_opt(1), x_opt(1)/(2*pi), x_opt(2)), ...
        'FontSize', 10, 'BackgroundColor', [0.95, 0.95, 0.95], 'EdgeColor', 'black');
end


%% fun_BSDA函数定义
function output = fun_BSDA(input)
    global N delta_t T k w_k S_yn;
    omiga = input(1);
    zeta = input(2);
    S_f0 = input(3);
    sigma_e = input(4);
    
    % 参数检查 - 添加下限约束
    if any(input < 0) || zeta >= 1 || sigma_e < 0.0001 || omiga < 0.1
        Result = Inf;
        output = Result;
    else
        % 确保zeta不会太接近1，以避免sqrt(1-zeta^2)接近零
        if zeta > 0.95
            zeta = 0.95;
        end
        
        % 计算自相关函数前先处理zeta特殊情况
        R = zeros(1, N);
        
        if zeta == 0
            % 特殊处理：当zeta=0时，避免除以零
            for n = 1:N-1
                tau = n * delta_t;
                R(n) = cos(omiga * tau) * pi * S_f0 / (2.0 * omiga^3);
            end
        else
            Omiga_d = sqrt(1 - zeta^2) * omiga;
            
            % 计算自相关函数
            for n = 1:N-1
                tau = n * delta_t;
                R(n) = (cos(Omiga_d * tau) + sin(Omiga_d * tau) * zeta / sqrt(1 - zeta^2)) * ...
                       exp(-zeta * omiga * tau) * pi * S_f0 / (2.0 * omiga^3 * zeta);
                
                % 添加数值稳定性检查
                if isnan(R(n)) || isinf(R(n))
                    R(n) = 0;
                end
            end
        end
        
        % 计算理论频谱
        temp3 = (sigma_e^2 * delta_t) / (2 * pi);
        temp4 = delta_t / (2 * pi * N);
        En = real(fft([N, 2*(N-1:-1:2).*R(2:N-1)]));
        E_initial = temp4 * En + temp3;
        
        % 添加数值稳定性检查
        for i = 1:length(E_initial)
            if E_initial(i) <= 0
                E_initial(i) = 1e-10;  % 防止除零
            end
        end
        
        % 计算似然函数
        p = (1:k);
        Result = sum(S_yn(1:k)./E_initial(2:k+1) + log(E_initial(2:k+1)));
        output = Result;
    end
end
