clc;
clear;
close all;

% 全局变量
global N delta_t T k w_k S_yn;
N = 120001;
T = 600;
delta_t = 0.005;
delta_tg = 0.0005;
omiga = 4;
zeta = 0.01;
S_f0 = 1;
T_add = 200;

% 开始模拟数据分析
disp('开始模拟数据分析...');

% 响应生成
N_g = (T_add + T) / delta_tg;
sigma_f = sqrt((2 * pi * S_f0) / delta_tg);
f = sigma_f * randn(1, N_g);
y = zeros(2, N_g);

% 状态空间模型
A = [0 1; -omiga^2 -2 * omiga * zeta];
B = [0 1]';

% 离散化
A_d = expm(A * delta_tg);
B_d = inv(A) * (A_d - eye(size(A))) * B;

% 生成响应
for n = 1:N_g-1
    y(:, n+1) = A_d * y(:, n) + B_d * f(n);
end

% 提取数据
xx = y(1, end - 10 * (N-1):10:end);
sigma_e = 0.1 * std(xx);
sigma_e2 = sigma_e^2;  % 修正：保存σ_e的平方值
noise = sigma_e * randn(1, N);
z = xx + noise;

% 计算频谱
k = 1000;
w_k = (2 * pi * (1:k)) / T;
part1 = zeros(1, k);
for n = 1:N-1
    part1 = part1 + z(n+1) * exp(-1i * w_k * n * delta_t);
end
part1 = part1 + z(1);
S_yn = (delta_t / (2 * pi * N)) * abs(part1).^2;

% PPT第11页: 频谱图
figure('Position', [100, 500, 800, 500]);
semilogy(w_k, S_yn(1:k), 'b-', 'LineWidth', 1.5);
grid on;
xlabel('w_k');
ylabel('S_{yn}(w_k)');
ylim([1e-14, 1e-2]);
xlim([0, 12]);
ax = gca;
ax.XGrid = 'on';
ax.YGrid = 'on';
ax.GridColor = [0.5, 0.5, 0.5];
ax.GridAlpha = 0.7;

% 在图中添加代码框
annotation('textbox', [0.02, 0.02, 0.28, 0.25], 'String', ...
    ['k = 1000;' newline ...
     'w_k = (2 * pi * (1:k)) / T;' newline ...
     'part1 = zeros(1, k);' newline ...
     'for n = 1:N-1' newline ...
     '    part1 = part1 + measurement(n+1) * exp(-1i * w_k * n * delta_t);' newline ...
     'end' newline ...
     'part1 = part1 + measurement(1);' newline ...
     'S_yn = (delta_t / (2 * pi * N)) * abs(part1).^2;' newline ...
     'S_log = log(S_yn);' newline ...
     'plot(w_k(1:1000), S_log(1:1000));' newline ...
     'xlabel(''w_k'');' newline ...
     'ylabel(''S_yn(w_k)'');' newline ...
     'grid on'], ...
    'FontSize', 9, ...
    'BackgroundColor', [0.95, 0.95, 0.95], 'EdgeColor', 'black');

% 在图上添加文字说明
annotation('textbox', [0.35, 0.80, 0.6, 0.1], 'String', ...
    'Using the measurement data, extract the first one thousand points. Then, execute the', ...
    'FontSize', 11, 'EdgeColor', 'none');
annotation('textbox', [0.35, 0.75, 0.6, 0.1], 'String', ...
    'following code to identify the spectrum, as depicted in the figure. The peak is', ...
    'FontSize', 11, 'EdgeColor', 'none');
annotation('textbox', [0.35, 0.70, 0.6, 0.1], 'String', ...
    'approximately located at Wk equal to 7.', ...
    'FontSize', 11, 'EdgeColor', 'none');

% 参数估计
x0 = [15, 1, 10, 1];
disp(['初始值: [' num2str(x0) ']']);

% 目标函数定义
fun = @fun_BSDA;

% 使用多步优化方法，逐渐缩小搜索范围
x_current = x0;

% 第一步：大步长搜索，快速离开起始点
disp(' ');
disp('第一步：大步长快速优化...');
% 先单独优化每个参数，确保函数值不会是Inf
for i = 1:4
    x_temp = x_current;
    
    % 针对每个参数使用不同的搜索范围
    if i == 1  % omega
        search_range = [2:0.5:8];
    elseif i == 2  % zeta
        search_range = [0.001:0.01:0.5];
    elseif i == 3  % S_f0
        search_range = [0.1:0.5:5];
    else  % sigma_e^2
        search_range = [0.001:0.01:0.5];
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

% 第二步：常规优化
disp(' ');
disp('第二步：精细优化...');
options = optimset('MaxFunEvals', 20000, 'MaxIter', 10000, 'TolFun', 1e-8, 'TolX', 1e-8);
[x_opt, fval, exitflag, output] = fminsearch(fun, x_current, options);

% 输出优化结果 - 修正格式
disp('优化结果:');
disp(['自然频率 (ω): ', num2str(x_opt(1)), ' rad/s']);
disp(['阻尼比 (ζ): ', num2str(x_opt(2))]);
disp(['白噪声谱强度 (S_f0): ', num2str(x_opt(3))]);
disp(['测量噪声方差 (σ_e^2): ', num2str(x_opt(4))]);

% 不确定性量化
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
COV = SD ./ abs(x_opt(:));  % 修正：将x_opt转换为列向量

% 计算归一化差异
actual_values = [omiga, zeta, S_f0, sigma_e2];  % 修正：使用sigma_e2而不是sigma_e
actual_values = actual_values(:);  % 转换为列向量
x_opt = x_opt(:);  % 转换为列向量
ND = abs(x_opt - actual_values) ./ SD;

% 正态分布PDF函数（替代normpdf，不需要统计工具箱）
normal_pdf = @(x, mu, sigma) (1./(sigma * sqrt(2*pi))) .* exp(-0.5 * ((x - mu)./sigma).^2);

% 高斯分布图（左上 - omega，右上 - zeta）
figure('Position', [100, 600, 1000, 400]);

% 第1个图：自然频率(ω)的高斯分布（左）
subplot(1,2,1);
mu1 = x_opt(1);
sigmaPDF1 = SD(1);
xxx = linspace(mu1 - 4*sigmaPDF1, mu1 + 4*sigmaPDF1, 1000);
yyy = normal_pdf(xxx, mu1, sigmaPDF1);
plot(xxx, yyy, 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410]);
grid on;
title('高斯分布曲线');
xlabel('x');
ylabel('概率密度');

% 阻尼比(ζ)的高斯分布
subplot(1,2,2);
mu2 = x_opt(2);
sigmaPDF2 = SD(2);
xxx = linspace(mu2 - 4*sigmaPDF2, mu2 + 4*sigmaPDF2, 1000);
yyy = normal_pdf(xxx, mu2, sigmaPDF2);
plot(xxx, yyy, 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410]);
grid on;
title('高斯分布曲线');
xlabel('x');
ylabel('概率密度');

% 高斯分布图（左 - S_f0，右 - sigma_e^2）
figure('Position', [100, 200, 1000, 400]);

% 白噪声谱强度(S_f0)的高斯分布
subplot(1,2,1);
mu3 = x_opt(3);
sigmaPDF3 = SD(3);
xxx = linspace(mu3 - 4*sigmaPDF3, mu3 + 4*sigmaPDF3, 1000);
yyy = normal_pdf(xxx, mu3, sigmaPDF3);
plot(xxx, yyy, 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410]);
grid on;
title('高斯分布曲线');
xlabel('x');
ylabel('概率密度');

% 测量噪声方差(σ_e^2)的高斯分布
subplot(1,2,2);
mu4 = x_opt(4);
sigmaPDF4 = SD(4);
xxx = linspace(max(0, mu4 - 4*sigmaPDF4), mu4 + 4*sigmaPDF4, 1000);
yyy = normal_pdf(xxx, mu4, sigmaPDF4);
plot(xxx, yyy, 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410]);
grid on;
title('高斯分布曲线');
xlabel('x');
ylabel('概率密度');

% 输出不确定性结果
disp(' ');
disp('不确定性量化结果:');
disp('参数         | 实际值     | 优化值     | 标准差(SD) | 变异系数(COV) | 归一化差异(ND)');
disp('-------------------------------------------------------------------------');

% 添加调试信息：检查变量
disp('Debug info - checking variables:');
disp(['omiga: ', num2str(omiga), ' (type: ', class(omiga), ')']);
disp(['zeta: ', num2str(zeta), ' (type: ', class(zeta), ')']);
disp(['S_f0: ', num2str(S_f0), ' (type: ', class(S_f0), ')']); 
disp(['sigma_e2: ', num2str(sigma_e2), ' (type: ', class(sigma_e2), ')']);
disp(['x_opt: ', num2str(x_opt(:)'), ' (type: ', class(x_opt), ')']);
disp(['SD: ', num2str(SD(:)'), ' (type: ', class(SD), ')']);
disp(['COV: ', num2str(COV(:)'), ' (type: ', class(COV), ')']);
disp(['ND: ', num2str(ND(:)'), ' (type: ', class(ND), ')']);
disp(' ');

% 直接格式化输出，避免使用数组索引
row1 = sprintf('自然频率 (ω)    | %.4f | %.4f | %.4f | %.4f | %.4f', ...
    omiga, x_opt(1), SD(1), COV(1), ND(1));
row2 = sprintf('阻尼比 (ζ)      | %.4f | %.4f | %.4f | %.4f | %.4f', ...
    zeta, x_opt(2), SD(2), COV(2), ND(2));
row3 = sprintf('白噪声谱强度 (S_f0) | %.4f | %.4f | %.4f | %.4f | %.4f', ...
    S_f0, x_opt(3), SD(3), COV(3), ND(3));
row4 = sprintf('测量噪声方差 (σ_e^2) | %.4f | %.4f | %.4f | %.4f | %.4f', ...
    sigma_e2, x_opt(4), SD(4), COV(4), ND(4));

disp(row1);
disp(row2);
disp(row3);
disp(row4);

% 测量频谱与理论频谱对比图（模拟数据）
figure('Position', [100, 50, 800, 500]);

% 绘制测量频谱
semilogy(w_k, S_yn(1:k), 'b-', 'LineWidth', 1.5);
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
    Omiga_d = sqrt(1 - zeta_opt^2) * omiga_opt;
    R(n) = (cos(Omiga_d * tau) + sin(Omiga_d * tau) * zeta_opt / sqrt(1 - zeta_opt^2)) * ...
           exp(-zeta_opt * omiga_opt * tau) * pi * S_f0_opt / (2.0 * omiga_opt^3 * zeta_opt);
end

temp3 = (sigma_e_opt^2 * delta_t) / (2 * pi);
temp4 = delta_t / (2 * pi * N);
En = real(fft([N, 2*(N-1:-1:2).*R(2:N-1)]));
E_theoretical = temp4 * En + temp3;

semilogy(w_k, E_theoretical(2:k+1), 'r--', 'LineWidth', 1.5);
grid on;
xlabel('w_k');
ylabel('S_{yn}(w_k)');
ylim([1e-14, 1e-2]);
xlim([0, 12]);
title('Measurement vs Simulation');
legend('Measured', 'Theoretical', 'Location', 'northeast');

% fun_BSDA函数定义
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
        
        Omiga_d = sqrt(1 - zeta^2) * omiga;
        R = zeros(1, N);
        
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