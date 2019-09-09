addpath(genpath('./dlt_cnn_map_dropout_nobiasnn'));
rand('state', 0);
load mnist_uint8;
load fnn;
train_x = double(train_x(1:20000, :)) / 255;
test_x  = double(test_x(1:1000,:))  / 255;
train_y = double(train_y(1:20000, :));
test_y  = double(test_y(1:1000,:));

%% Noise 
noise_type = 'none';

if strcmp(noise_type, 'none') == 1
    % SFNN-IP (2000Hz):97.2%, 4ms, SFNN-noIP (2000Hz):96.7%, 16ms
    % SFNN-IP (200Hz):97.0%, 20ms, SFNN-noIP (200Hz): 96.3%, 78ms
    % SFNN-IP (20Hz): <20%,        SFNN-noIP (20Hz):<20%
    % Analog FNN: 97.0%
    train_noise = zeros(size(train_x));
    test_noise = zeros(size(test_x));
elseif strcmp(noise_type, 'gaussian') == 1  
    % SFNN-IP (2000Hz):51.3%, --ms, SFNN-noIP (2000Hz):40.8%, --ms
    % SFNN-IP (200Hz):84.8%, --ms,  SFNN-noIP (200Hz): 77.1%, --ms
    % SFNN-IP (20Hz): 73.6%, --ms,  SFNN-noIP (20Hz):37.6%
    % Analog FNN: 93.1%
    a = 0;
    b = 0.33465;
    train_noise = a + b * randn(size(train_x));
    test_noise = a + b * randn(size(test_x));
elseif strcmp(noise_type, 'rayleigh') == 1
    % SFNN-IP (2000Hz):45.4%, --ms, SFNN-noIP (2000Hz):34.4%, --ms
    % SFNN-IP (200Hz):82.7%, --ms,  SFNN-noIP (200Hz): 72.7%, --ms
    % SFNN-IP (20Hz): 71.3%, --ms,  SFNN-noIP (20Hz):31.2%
    % Analog FNN: 95.1%
    a = 0;
    b = 0.33417;
    train_noise = a + b * (-log(1-rand(size(train_x)))).^0.5 .* sign(rand(size(train_x))-0.5);
    test_noise = a + b * (-log(1-rand(size(test_x)))).^0.5 .* sign(rand(size(test_x))-0.5);
elseif strcmp(noise_type, 'uniform') == 1
    % SFNN-IP (2000Hz):46.5%, --ms, SFNN-noIP (2000Hz):34.9%, --ms
    % SFNN-IP (200Hz):82.9%, --ms,  SFNN-noIP (200Hz): 73.9%, --ms
    % SFNN-IP (20Hz): 71.7%, --ms,  SFNN-noIP (20Hz):33.8%
    % Analog FNN: 93.4%
    a = 0;
    b = 0.5793;
    train_noise =  a + (b - a) * (rand(size(train_x))-0.5) * 2;
    test_noise =  a + (b - a) * (rand(size(test_x))-0.5) * 2;
elseif strcmp(noise_type, 'gamma') == 1
    % SFNN-IP (2000Hz):24.9%, --ms, SFNN-noIP (2000Hz):<20%, --ms
    % SFNN-IP (200Hz):73.4%, --ms,  SFNN-noIP (200Hz): 63.6%, --ms
    % SFNN-IP (20Hz): 62.2%, --ms,  SFNN-noIP (20Hz):<20%
    % Analog FNN: 63.2%
    a = 18.78;
    b = 4;
    train_noise = zeros(size(train_x));
    test_noise = zeros(size(test_x));
    for i = 1 : b
        train_noise = train_noise + (-1/a) .* log(1 - rand(size(train_noise)));
        test_noise = test_noise + (-1/a) .* log(1 - rand(size(test_noise)));
    end
elseif strcmp(noise_type, 'pepper') == 1
    % SFNN-IP (2000Hz):96.1%, 2ms, SFNN-noIP (2000Hz):95.2%, 7ms
    % SFNN-IP (200Hz):95.3%, 12ms, SFNN-noIP (200Hz): 94.0%, 26ms
    % SFNN-IP (20Hz): 85.6%, --ms, SFNN-noIP (20Hz):<20%
    % Analog FNN: 94.9%
    % 5%-->96.1%, 10% --> 95.00%, 20% --> 89.5%, 30%-->77.2%, 50%-->55.8%
    % Data_norm: 50%-->54.7%, 30%-->77.1%, 
    a = 0.1;
    b = 0.1;
    x = rand(size(train_x));
    train_noise = zeros(size(train_x));
    train_noise(x<=a) = 0;
    train_noise(x>a & x<(a+b)) = 1;
    t_x = rand(size(test_x));
    test_noise = zeros(size(test_x));
    test_noise(t_x<=a) = 0;
    test_noise(t_x>a & t_x<(a+b)) = 1;
elseif strcmp(noise_type, 'light-') == 1
    % SFNN-IP (2000Hz):95.7%, 2ms, SFNN-noIP (2000Hz):96.2%, 6ms
    % SFNN-IP (200Hz):95.4%, 18ms, SFNN-noIP (200Hz): 95.7%, 44ms
    % SFNN-IP (20Hz): 78.0%,        SFNN-noIP (20Hz):<20%
    % Analog FNN: 54.90%
    amp = -0.4915;
    train_noise = ones(size(train_x)) * amp;
    test_noise = ones(size(test_x)) * amp;
elseif strcmp(noise_type, 'light+') == 1
    % SFNN-IP (2000Hz):21.4%, --ms, SFNN-noIP (2000Hz):16.8%, --ms
    % SFNN-IP (200Hz):70.3%, --ms, SFNN-noIP (200Hz): 61.4%, --ms
    % SFNN-IP (20Hz): 58.6%,        SFNN-noIP (20Hz):<20%
    % Analog FNN: 59.90%
    amp = 0.2283;
    train_noise = ones(size(train_x)) * amp;
    test_noise = ones(size(test_x)) * amp;
end

noisy_train_x = train_x + train_noise;
noisy_test_x = test_x + test_noise;
test_x(test_x > 1) = 1;
test_x(test_x < 0) = 0;

%% Digits show 
nim = reshape(noisy_test_x([4,6,13,17],:),4, 28, 28);
h = figure;
subplot(1,3,1); imshow(reshape(noisy_test_x(4,:),28, 28)');
subplot(1,3,2); imshow(reshape(noisy_test_x(6,:),28, 28)');
subplot(1,3,3); imshow(reshape(noisy_test_x(13,:),28, 28)');
set(h,'position',[100 100 500 150]);

%% Parameter Setting
t_opts = struct;
t_opts.t_ref        = 0.000;
t_opts.threshold    =  1;
t_opts.dt           = 0.001;
t_opts.duration     = 0.100;
t_opts.report_every = 0.001;
t_opts.max_rate     =   200;

%% Normalization Method
% [data_norm_fnn, ~] = normalize_nn_data(fnn, train_x);
% data_norm_fnn = nnlifsim(data_norm_fnn, noisy_test_x, test_y, t_opts);

% [model_norm_fnn, norm_constants] = normalize_nn_model(fnn);
% model_norm_fnn = nnlifsim(model_norm_fnn, noisy_test_x, test_y, t_opts);
% fprintf('NN normalized.\n');

%% Run SFNN
ip_opts.tau_ip = 1000; ip_opts.beta = 0.6; ip_opts.eta = 0.5; ip_opts.initial_rR = 10; ip_opts.initial_rC = 10;

t_opts.max_rate     =   20;
nn_nonip1 = nnlifsim(fnn, noisy_test_x, test_y, t_opts);
nn_ip1 = nnlifsim_ip(fnn, noisy_test_x, test_y, t_opts, ip_opts);
t_opts.max_rate     =   200;
nn_nonip2 = nnlifsim(fnn, noisy_test_x, test_y, t_opts);
nn_ip2 = nnlifsim_ip(fnn, noisy_test_x, test_y, t_opts, ip_opts);
t_opts.max_rate     =   2000;
nn_nonip3 = nnlifsim(fnn, noisy_test_x, test_y, t_opts);
nn_ip3 = nnlifsim_ip(fnn, noisy_test_x, test_y, t_opts, ip_opts);

%% 
% figure;
% plot(t_opts.dt:t_opts.dt:t_opts.duration, nn_nonip1.performance,'r--');
% hold on;
% plot(t_opts.dt:t_opts.dt:t_opts.duration, nn_ip1.performance,'b--');
% hold on;
% plot(t_opts.dt:t_opts.dt:t_opts.duration, nn_nonip2.performance,'r');
% hold on;
% plot(t_opts.dt:t_opts.dt:t_opts.duration, nn_ip2.performance,'b');
%% Run FNN
[er, bad] = nntest(fnn, noisy_test_x, test_y);
fprintf('Test Accuracy: %2.2f%%.\n', (1-er)*100);

% Draw Figure
figure;
plot(t_opts.dt:t_opts.dt:t_opts.duration, nn_nonip1.performance,'r--');
hold on;
plot(t_opts.dt:t_opts.dt:t_opts.duration, nn_ip1.performance,'b--');
hold on;
plot(t_opts.dt:t_opts.dt:t_opts.duration, nn_nonip2.performance,'r');
hold on;
plot(t_opts.dt:t_opts.dt:t_opts.duration, nn_ip2.performance,'b');
hold on;
plot(t_opts.dt:t_opts.dt:t_opts.duration, nn_nonip3.performance,'y');
hold on;
plot(t_opts.dt:t_opts.dt:t_opts.duration, nn_ip3.performance,'g');
grid;
legend('SFNN-noIP(20Hz)', 'SFNN-IP(20Hz)','SFNN-noIP(200Hz)', 'SFNN-IP(200Hz)','SFNN-noIP(2000Hz)', 'SFNN-IP(2000Hz)');
% ylim([90 100]);
xlabel('Time [s]');
ylabel('Accuracy [%]');