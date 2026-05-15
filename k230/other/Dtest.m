% 1. 输入你刚采集的实际距离数据 D (单位: cm)
D_true = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100];

% 2. 输入对应的 K230 像素坐标 Y
Y_pixel = [440, 414, 395, 383, 368, 354, 342, 335, 330, 320, 312];

% 3. 使用二阶多项式拟合: D = a*Y^2 + b*Y + c
p = polyfit(Y_pixel, D_true, 3);

% 打印拟合系数，准备写死到 K230 代码里
fprintf('===== 请将以下系数复制到你的 Python 脚本中 =====\n');
fprintf('a = %.8f\n', p(1)); % 三阶项系数 (x^3)
fprintf('b = %.8f\n', p(2)); % 二阶项系数 (x^2)
fprintf('c = %.8f\n', p(3)); % 一阶项系数 (x)
fprintf('d = %.8f\n', p(4)); % 常数项
fprintf('================================================\n');

% 4. 计算拟合误差 (RMSE) 评估精度
D_predict = polyval(p, Y_pixel);
RMSE = sqrt(mean((D_predict - D_true).^2));
fprintf('当前标定数据的均方根误差 (RMSE) = %.4f cm\n', RMSE);

% 计算每个测试点的具体误差绝对值
max_error = max(abs(D_predict - D_true));
fprintf('最大绝对误差 = %.4f cm\n', max_error);

% 5. 绘制拟合曲线进行可视化比对
figure;
plot(Y_pixel, D_true, 'ro', 'MarkerSize', 8, 'LineWidth', 2); hold on;
Y_dense = linspace(min(Y_pixel), max(Y_pixel), 100);
D_dense = polyval(p, Y_dense);
plot(Y_dense, D_dense, 'b-', 'LineWidth', 2);
xlabel('像素坐标 Y (Pixel)');
ylabel('实际距离 D (cm)');
title('像素坐标与距离的二阶回归拟合');
legend('实测数据点', '二次拟合曲线');
grid on;