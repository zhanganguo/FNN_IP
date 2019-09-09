%% Train an example ConvNet to achieve very high classification, fast.
%    Load paths
addpath(genpath('./dlt_cnn_map_dropout_nobiasnn'));
%% Load data
rand('state', 11);
load mnist_uint8;
train_x = double(reshape(train_x',28,28,60000)) / 255;
test_x = double(reshape(test_x',28,28,10000)) / 255;
train_y = double(train_y');
test_y = double(test_y');
%% Noise 
noise_type = 'gaussian';

if strcmp(noise_type, 'none') == 1
    train_noise = zeros(size(train_x));
    test_noise = zeros(size(test_x));
elseif strcmp(noise_type, 'gaussian') == 1  
    a = 0;
    b = 0.5;
    train_noise = a + b * randn(size(train_x));
    test_noise = a + b * randn(size(test_x));
elseif strcmp(noise_type, 'rayleigh') == 1
    a = 0;
    b = 0.5;
    train_noise = a + b * (-log(1-rand(size(train_x)))).^0.5 .* sign(rand(size(train_x))-0.5);
    test_noise = a + b * (-log(1-rand(size(test_x)))).^0.5 .* sign(rand(size(test_x))-0.5);
elseif strcmp(noise_type, 'uniform') == 1
    a = 0;
    b = 0.5;
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

train_x = train_x + train_noise;
test_x = test_x + test_noise;
% test_x = test_x - 0.25;
% test_x(test_x > 1) = 1;

%% Initialize net
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 16, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 16, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};
cnn = cnnsetup(cnn, train_x, train_y);
% Set the activation function to be a ReLU
cnn.act_fun = @(inp)max(0, inp);
% Set the derivative to be the binary derivative of a ReLU
cnn.d_act_fun = @(forward_act)double(forward_act>0);
%% ReLU Train
% Set up learning constants
opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs =  5;    
opts.learn_bias = 0;
opts.dropout = 0.0;
cnn.first_layer_dropout = 0;
% Train - takes about 199 seconds per epoch on my machine

cnn = cnntrain(cnn, train_x, train_y, opts);
% Test
[er, bad] = cnntest(cnn, train_x, train_y);
fprintf('TRAINING Accuracy: %2.2f%%.\n', (1-er)*100);
[er, bad] = cnntest(cnn, test_x, test_y);
fprintf('Test Accuracy: %2.2f%%.\n', (1-er)*100);
%% Spike-based Testing of a ConvNet
t_opts = struct;
t_opts.t_ref        = 0.000;
t_opts.threshold    =   1.0;
t_opts.dt           = 0.001;
t_opts.duration     = 0.060;
t_opts.report_every = 0.001;
t_opts.max_rate     =  200;
cnn = convlifsim(cnn, test_x, test_y, t_opts);
fprintf('Done.\n');
%% Data-normalize the CNN
% [norm_convnet, norm_constants] = normalize_cnn_data(cnn, train_x);
% for idx=1:numel(norm_constants)
%     fprintf('Normali+zation Factor for Layer %i: %3.5f\n',idx, norm_constants(idx));
% end
% fprintf('ConvNet normalized.\n');
%% Test the Data-Normalized CNN
% t_opts = struct;
% t_opts.t_ref        = 0.000;
% t_opts.threshold    =   1.0;
% t_opts.dt           = 0.001;
% t_opts.duration     = 0.030;
% t_opts.report_every = 0.001;
% t_opts.max_rate     =  1000;
% norm_convnet = convlifsim(norm_convnet, test_x, test_y, t_opts);
% fprintf('Done.\n');
%% Show the difference
figure(1); clf;
% plot(t_opts.dt:t_opts.dt:t_opts.duration, norm_convnet.performance,'r');
% hold on; grid on;
plot(t_opts.dt:t_opts.dt:t_opts.duration, cnn.performance,'b');
legend('Normalized ConvNet, Default Params');
ylim([00 100]);
xlabel('Time [s]');
ylabel('Accuracy [%]');