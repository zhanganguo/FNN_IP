function nn=nnlifsim_ip(nn, test_x, test_y, opts, ip_opts)
dt = opts.dt;
nn.performance = [];
num_examples = size(test_x,1);

tau_ip = ip_opts.tau_ip;
beta = ip_opts.beta;
eta = ip_opts.eta;
initial_rC = ip_opts.initial_rC;
initial_rR = ip_opts.initial_rR;

% Initialize network architecture
for l = 1 : numel(nn.size)
    blank_neurons = zeros(num_examples, nn.size(l));
    one_neurons = ones(num_examples, nn.size(l));
    nn.layers{l}.mem = blank_neurons;
    nn.layers{l}.refrac_end = blank_neurons;
    nn.layers{l}.sum_spikes = blank_neurons;

    nn.layers{l}.rC = one_neurons * initial_rC;
    nn.layers{l}.rR = one_neurons * initial_rR;

end

% Precache answers
[~,   ans_idx] = max(test_y');

for t=dt:dt:opts.duration
    % Create poisson distributed spikes from the input images
    %   (for all images in parallel)
    rescale_fac = 1/(dt*opts.max_rate);
    spike_snapshot = rand(size(test_x)) * rescale_fac;
    inp_image = spike_snapshot <= test_x;
    
    nn.layers{1}.spikes = inp_image;
    nn.layers{1}.sum_spikes = nn.layers{1}.sum_spikes + inp_image;
    for l = 2 : numel(nn.size)
        % Get input impulse from incoming spikes
        I = nn.layers{l-1}.spikes * nn.W{l-1}';
        
        Cm = 1./nn.layers{l}.rC;
        Rm = 1./nn.layers{l}.rR;
        tau_m = Rm.* Cm;
        dv = Rm.* I./ tau_m * dt * (1/dt);
        
        % Add input to membrane p otential
        nn.layers{l}.mem = nn.layers{l}.mem + dv;
        % Check for spiking
        nn.layers{l}.spikes = nn.layers{l}.mem >= opts.threshold;
        % Reset
        nn.layers{l}.mem(nn.layers{l}.spikes) = 0;
        % Ban updates until....
        nn.layers{l}.refrac_end(nn.layers{l}.spikes) = t + opts.t_ref;
        % Store result for analysis later
        nn.layers{l}.sum_spikes = nn.layers{l}.sum_spikes + nn.layers{l}.spikes;
        
        % y = sigma * nn.layers{l}.spikes;
        [size1,~] = size(nn.layers{l}.spikes);
        [~, size2] = size(nn.W{l-1});
        A = ones(size1,size2);
        y = nn.layers{l}.spikes .* (A * nn.W{l-1}') * eta;

        delta_rC = (1./nn.layers{l}.rC - y.*I + beta.*(1-y).*I)./tau_ip;
        delta_rR = (-nn.layers{l}.rR + y - beta.*(1-y))./tau_ip; 
        nn.layers{l}.rC = nn.layers{l}.rC + delta_rC;
        nn.layers{l}.rR = nn.layers{l}.rR + delta_rR;
    end
    
    if(mod(round(t/dt),round(opts.report_every/dt)) == round(opts.report_every/dt)-1)
        [~, guess_idx] = max(nn.layers{end}.sum_spikes');
        acc = sum(guess_idx==ans_idx)/size(test_y,1)*100;
        fprintf('Time: %1.3fs | Accuracy: %2.2f%%.\n', t, acc);
        nn.performance(end+1) = acc;
    else
        fprintf('.');
    end
end


% Get answer
[~, guess_idx] = max(nn.layers{end}.sum_spikes');
acc = sum(guess_idx==ans_idx)/size(test_y,1)*100;
fprintf('\nFinal spiking accuracy: %2.2f%%\n', acc);

end


