dbstop if error
%% Train an example FC network to achieve very high classification, fast.
%    Load paths
addpath(genpath('./dlt_cnn_map_dropout_nobiasnn'));
%% Load data
rand('state', 0);
load mnist_uint8;

train_x = double(train_x) / 255;
train_y = double(train_y);
test_x = double(test_x) / 255;
test_y = double(test_y);

%% Noise 
noise_type = 'none';

if strcmp(noise_type, 'none') == 1
    train_noise = zeros(size(train_x));
    test_noise = zeros(size(test_x));
elseif strcmp(noise_type, 'gaussian') == 1  
    a = 0;
    b = 0.33465;
    train_noise = a + b * randn(size(train_x));
    test_noise = a + b * randn(size(test_x));
elseif strcmp(noise_type, 'rayleigh') == 1
    a = 0;
    b = 0.5;
    train_noise = a + b * (-log(1-rand(size(train_x)))).^0.5 .* sign(rand(size(train_x))-0.5);
    test_noise = a + b * (-log(1-rand(size(test_x)))).^0.5 .* sign(rand(size(test_x))-0.5);
elseif strcmp(noise_type, 'uniform') == 1
    a = 0;
    b = 0.5793;
    train_noise =  a + (b - a) * (rand(size(train_x))-0.5) * 2;
    test_noise =  a + (b - a) * (rand(size(test_x))-0.5) * 2;
elseif strcmp(noise_type, 'gamma') == 1
    a = 9;
    b = 3;
    train_noise = zeros(size(train_x));
    test_noise = zeros(size(test_x));
    for i = 1 : b
        train_noise = train_noise + (-1/a) .* log(1 - rand(size(train_noise)));
        test_noise = test_noise + (-1/a) .* log(1 - rand(size(test_noise)));
    end
elseif strcmp(noise_type, 'pepper') == 1
    a = 0.25;
    b = 0.25;
    x = rand(size(train_x));
    train_noise = zeros(size(train_x));
    train_noise(x<=a) = 0;
    train_noise(x>a & x<(a+b)) = 1;
    t_x = rand(size(test_x));
    test_noise = zeros(size(test_x));
    test_noise(t_x<=a) = 0;
    test_noise(t_x>a & t_x<(a+b)) = 1;
end

noisy_train_x = train_x + train_noise;
noisy_test_x = test_x + test_noise;

noisy_train_x = train_x + train_noise;
noisy_train_x(noisy_train_x < 0) = 0;
noisy_train_x(noisy_train_x > 1) = 1;

noisy_test_x = test_x + test_noise;
noisy_test_x(noisy_test_x < 0) = 0;
noisy_test_x(noisy_test_x > 1) = 1;

%% Initialize net
% nn = nnsetup([784 1200 1200 800 800 10]);
% % Rescale weights for ReLU
% for i = 2 : nn.n   
%     % Weights - choose between [-0.1 0.1]
%     nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)) - 0.5) * 0.01 * 2;
%     nn.vW{i - 1} = zeros(size(nn.W{i-1}));
% end
% % Set up learning constants
% nn.activation_function = 'relu';
% nn.output ='relu';
% nn.learningRate = 1;
% nn.momentum = 0.5;
% nn.dropoutFraction = 0.5;
% nn.learn_bias = 0;
% opts.numepochs =  200;
% opts.batchsize = 100;
% % Train - takes about 15 seconds per epoch on my machine
% nn = nntrain(nn, noisy_train_x, train_y, opts);
% % Test - should be 98.62% after 15 epochs
% [er, train_bad] = nntest(nn, noisy_train_x, train_y);
% fprintf('TRAINING Accuracy: %2.2f%%.\n', (1-er)*100);

%% Directly Load fnn.mat
load fnn_2hidden;

%% Test Accuracy
[er, bad] = nntest(nn, noisy_test_x, test_y);
fprintf('Test Accuracy: %2.2f%%.\n', (1-er)*100);

%% Spike-based Testing of Fully-Connected NN
t_opts = struct;
t_opts.t_ref        = 0.000;
t_opts.threshold    =  1;
t_opts.dt           = 0.001;
t_opts.duration     = 0.100;
t_opts.report_every = 0.001;
t_opts.max_rate     =   200;

nn_nonip = nnlifsim(nn, noisy_test_x, test_y, t_opts);

ip_opts.tau_ip = 20;
ip_opts.beta = 0.6;
ip_opts.eta = 0.5;
ip_opts.initial_rC = 1;
ip_opts.initial_rR = 1;
nn_ip = nnlifsim_ip(nn, noisy_test_x, test_y, t_opts, ip_opts);
fprintf('Done.\n');

%% Data-normalize the NN
% [data_norm_nn, norm_constants] = normalize_nn_data(nn, train_x);
% data_norm_nn = nnlifsim(data_norm_nn, test_x, test_y, t_opts);

% [model_norm_nn, norm_constants] = normalize_nn_model(nn);
% model_norm_nn = nnlifsim(model_norm_nn, test_x, test_y, t_opts);
% fprintf('NN normalized.\n');

%% Test the Data-Normalized NN
% t_opts = struct;
% t_opts.t_ref        = 0.000;
% t_opts.threshold    =   1;
% t_opts.dt           = 0.001;
% t_opts.duration     = 0.050;
% t_opts.report_every = 0.001;
% t_opts.max_rate     =  1000;  
% t_opts.max_rate     =  200;
% norm_nn = nnlifsim(norm_nn, test_x, test_y, t_opts);
% fprintf('Done.\n');

%% Show the difference
figure;
plot(t_opts.dt:t_opts.dt:t_opts.duration, nn_nonip.performance,'r');
% hold on;
% plot(t_opts.dt:t_opts.dt:t_opts.duration, data_norm_nn.performance,'black');
% hold on;
% plot(t_opts.dt:t_opts.dt:t_opts.duration, model_norm_nn.performance,'m');
hold on;
plot(t_opts.dt:t_opts.dt:t_opts.duration, nn_ip.performance,'b');
grid;
legend('SFNN-noIP', 'SFNN-IP');
% ylim([90 100]);
xlabel('Time [s]');
ylabel('Accuracy [%]');
