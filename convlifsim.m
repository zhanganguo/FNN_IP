function cnn = convlifsim(cnn, test_x, test_y, opts)
num_examples = size(test_x, 3);
num_classes  = size(test_y, 1);
% Initialize a neuron-based network - needs to be activated to get all the
%   sizes. Shouldn't be an issue after training, unless cleaned.
for l = 1 : numel(cnn.layers)
    outputmaps = numel(cnn.layers{l}.a);
    cnn.layers{l}.mem = cell(1, outputmaps);
    % ---- Modification Start ---- %
    cnn.layers{l}.rC = cell(1, outputmaps);
    cnn.layers{l}.rR = cell(1, outputmaps);
    cnn.layers{l}.fired = cell(1, outputmaps);
    cnn.layers{l}.spiking = cell(1, outputmaps);
    spike = cell(1,4);
    % ---- Modification End ---- %
    for j=1:outputmaps
        correctly_sized_zeros = zeros(size(cnn.layers{l}.a{j}, 1), ...
            size(cnn.layers{l}.a{j}, 2), num_examples);
        cnn.layers{l}.mem{j} = correctly_sized_zeros;
        cnn.layers{l}.refrac_end{j} = correctly_sized_zeros;        
        cnn.layers{l}.sum_spikes{j} = correctly_sized_zeros;
        % ---- Modification Start ---- %
        correctly_sized_ones = ones(size(cnn.layers{l}.a{j}, 1), ...
            size(cnn.layers{l}.a{j}, 2), num_examples);
        cnn.layers{l}.rC{j} = correctly_sized_ones * 2;
        cnn.layers{l}.rR{j} = correctly_sized_ones * 2;
        cnn.layers{l}.fired{j} = correctly_sized_zeros;
        [x,y,z] = size(correctly_sized_zeros);
%         fprintf('correctly_sized_zeros :x=%d,y=%d,z=%d\n',x,y,z);
        cnn.layers{l}.spiking{j} = zeros(ceil(opts.duration/opts.dt),x*y);
        % ---- Modification End ---- %
    end   
    % ---- Modification Start ---- %
    for lay = 2 : numel(cnn.layers)
        len = size(cnn.layers{lay}.a{1}, 1) * size(cnn.layers{lay}.a{1}, 2) * num_examples;
        spike{lay-1} = zeros(len, ceil(opts.duration/opts.dt));
    end
    % ---- Modification End ---- %
end
cnn.sum_fv = zeros(size(cnn.ffW,2), num_examples);
cnn.o_mem        = zeros(num_classes, num_examples);
cnn.o_refrac_end = zeros(num_classes, num_examples);
cnn.o_sum_spikes = zeros(num_classes, num_examples);
cnn.performance  = [];
% Precache answers
[~, ans_idx] = max(test_y);

       % ---- Modification Start ---- %
        len = length(0:opts.dt:opts.duration);
        Cm_his = zeros(numel(cnn.layers),len);
        Rm_his = zeros(numel(cnn.layers),len);
        index1 = 1;
        index2 = 1;
        % ---- Modification End ---- %
        
for t = 0:opts.dt:opts.duration
    % Create poisson distributed spikes from the input images (for all images in parallel)
    rescale_fac = 1/(opts.dt*opts.max_rate);
    spike_snapshot = rand(size(test_x)) * rescale_fac;
    inp_image = spike_snapshot <= test_x;

    cnn.layers{1}.spikes{1} = inp_image;
    cnn.layers{1}.mem{1} = cnn.layers{1}.mem{1} + inp_image;
    cnn.layers{1}.sum_spikes{1} = cnn.layers{1}.sum_spikes{1} + inp_image;
    inputmaps = 1;
    for l = 2 : numel(cnn.layers)   %  for each layer
        if strcmp(cnn.layers{l}.type, 'c')
            % Convolution layer, output a map for each convolution
            for j = 1 : cnn.layers{l}.outputmaps
                % Sum up input maps
                z = zeros(size(cnn.layers{l - 1}.spikes{1}) - [cnn.layers{l}.kernelsize - 1 cnn.layers{l}.kernelsize - 1 0]);
                for i = 1 : inputmaps   %  for each input map
                    %  convolve with corresponding kernel and add to temp output map
                    z = z + convn(cnn.layers{l - 1}.spikes{i}, cnn.layers{l}.k{i}{j}, 'valid');
                end
                
                % ---- Modification Start ---- %
                I = z;
                tau_ip = 1;
                Cm = 2./cnn.layers{l}.rC{j};
                Rm = 2./cnn.layers{l}.rR{j};
                tau_m = Rm.* Cm;
                sigma = 0.5;
                beta = 0.6;
                dv = Rm.* I./ tau_m;
%                 z = dv;
                if j==1
                    Cm_his(l,index1) = Cm(1,1);
                    Rm_his(l,index1) = Rm(1,1);
                end
                % ---- Modification End ---- %
                
                z(cnn.layers{l}.refrac_end{j} > t) = 0;                                                                                                                                      
                cnn.layers{l}.mem{j} = cnn.layers{l}.mem{j} + z;
                cnn.layers{l}.spikes{j} = cnn.layers{l}.mem{j} >= opts.threshold;
                if j==1
                    [size_x,size_y] = size(cnn.layers{l}.spikes{j});
                    for s = 1:size_x
                        cnn.layers{l}.spiking{j}(index1, (s-1)*size_x+1:s*size_x) = cnn.layers{l}.spikes{j}(:,s)';             
                    end
                    lens = size(cnn.layers{l}.spikes{1},1) * size(cnn.layers{l}.spikes{1},2) * size(cnn.layers{l}.spikes{1},3);
                    spike{l-1}(:,index1) = reshape(cnn.layers{l}.spikes{1}, 1, lens);
                    index1 = index1 + 1;
                end
                cnn.layers{l}.mem{j}(cnn.layers{l}.spikes{j}) = 0;
                cnn.layers{l}.refrac_end{j}(cnn.layers{l}.spikes{j}) = t + opts.t_ref;
                cnn.layers{l}.sum_spikes{j} = cnn.layers{l}.sum_spikes{j} + cnn.layers{l}.spikes{j};
                
                % ---- Modification Start ---- %
                cnn.layers{l}.fired{j} = cnn.layers{l}.refrac_end{j} == t + opts.t_ref;
                y = sigma * cnn.layers{l}.fired{j};
                delta_rC = (1./cnn.layers{l}.rC{j} - y.*I + beta.*(1-y).*I)./tau_ip.*opts.dt;
                delta_rR = (-cnn.layers{l}.rR{j} + y - beta.*(1-y))./tau_ip.*opts.dt;
                cnn.layers{l}.rC{j} = cnn.layers{l}.rC{j} + delta_rC;
                cnn.layers{l}.rR{j} = cnn.layers{l}.rR{j} + delta_rR;
                % ---- Modification End ---- %
            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = cnn.layers{l}.outputmaps;
                
        elseif strcmp(cnn.layers{l}.type, 's')
            %  Subsample by averaging
            for j = 1 : inputmaps
                % Average input
                z = convn(cnn.layers{l - 1}.spikes{j}, ones(cnn.layers{l}.scale) / (cnn.layers{l}.scale ^ 2), 'valid');
                % Downsample
                z = z(1 : cnn.layers{l}.scale : end, 1 : cnn.layers{l}.scale : end, :);
                
                % ---- Modification Start ---- %
                I = z;
                Cm = 1./cnn.layers{l}.rC{j};
                Rm = 1./cnn.layers{l}.rR{j};
                tau_m = Rm.* Cm;
%                 delta_z = (Vrest - cnn.layers{l}.mem{j} + Rm.*I)./tau_m .* opts.dt * theta;
%                 dv = Rm.* I ./ tau_m .* opts.dt;
                dv = Rm.* I ./ tau_m;
%                 z = dv;
                
                if j==1
                    Cm_his(l,index2) = Cm(1,1);
                    Rm_his(l,index2) = Rm(1,1);
                end
                % ---- Modification End ---- %
                
                z(cnn.layers{l}.refrac_end{j} > t) = 0;
                cnn.layers{l}.mem{j} = cnn.layers{l}.mem{j} + z;
                cnn.layers{l}.spikes{j} = cnn.layers{l}.mem{j} >= opts.threshold;
                if j==1
                    [size_x,size_y] = size(cnn.layers{l}.spikes{j});
                    for s = 1:size_x
                        cnn.layers{l}.spiking{j}(index1, (s-1)*size_x+1:s*size_x) = cnn.layers{l}.spikes{j}(:,s)';
                    end
                    lens = size(cnn.layers{l}.spikes{1},1) * size(cnn.layers{l}.spikes{1},2) * size(cnn.layers{l}.spikes{1},3);
                    spike{l-1}(:,index2) = reshape(cnn.layers{l}.spikes{1}, 1, lens);
                    index2 = index2 + 1;
                end
                cnn.layers{l}.mem{j}(cnn.layers{l}.spikes{j}) = 0;
                cnn.layers{l}.refrac_end{j}(cnn.layers{l}.spikes{j}) = t + opts.t_ref;              
                cnn.layers{l}.sum_spikes{j} = cnn.layers{l}.sum_spikes{j} + cnn.layers{l}.spikes{j};    
                
                % ---- Modification Start ---- %
                cnn.layers{l}.fired{j} = cnn.layers{l}.refrac_end{j} == t + opts.t_ref;
                y = sigma * cnn.layers{l}.fired{j};
                delta_rC = (1./cnn.layers{l}.rC{j} - y.*I + beta.*(1-y).*I)./tau_ip.*opts.dt * 100;
                delta_rR = (-cnn.layers{l}.rR{j} + y - beta.*(1-y))./tau_ip.*opts.dt * 100;
                cnn.layers{l}.rC{j} = cnn.layers{l}.rC{j} + delta_rC;
                cnn.layers{l}.rR{j} = cnn.layers{l}.rR{j} + delta_rR;
                % ---- Modification End ---- %
            end
        end
    end
    
    % Concatenate all end layer feature maps into vector
    cnn.fv = [];
    for j = 1 : numel(cnn.layers{end}.spikes)
        sa = size(cnn.layers{end}.spikes{j});
        cnn.fv = [cnn.fv; reshape(cnn.layers{end}.spikes{j}, sa(1) * sa(2), sa(3))];
    end
    cnn.sum_fv = cnn.sum_fv + cnn.fv;
    
    % Run the output layer neurons
    %   Add inputs multiplied by weight
    impulse = cnn.ffW * cnn.fv;
    %   Only add input from neurons past their refractory point
    impulse(cnn.o_refrac_end >= t) = 0;
    %   Add input to membrane potential
    cnn.o_mem = cnn.o_mem + impulse;
    %   Check for spiking
    cnn.o_spikes = cnn.o_mem >= opts.threshold;
    %   Reset
    cnn.o_mem(cnn.o_spikes) = 0;
    %   Ban updates until....
    cnn.o_refrac_end(cnn.o_spikes) = t + opts.t_ref;
    %   Store result for analysis later
    cnn.o_sum_spikes = cnn.o_sum_spikes + cnn.o_spikes;
    
    % Tell the user what's going on
    if(mod(round(t/opts.dt),round(opts.report_every/opts.dt)) == ...
            0 && (t/opts.dt > 0))
        [~, guess_idx] = max(cnn.o_sum_spikes);
        acc = sum(guess_idx==ans_idx)/size(test_y, 2) * 100;
        fprintf('Time: %1.3fs | Accuracy: %2.2f%%.\n', t, acc);
        cnn.performance(end+1) = acc;
    else
        fprintf('.');            
    end
end

%     figure;
%     plot(1:length(Cm_his(1,:)), Cm_his(1,:));
%     title('Cm of layer 1');
%     figure;
%     plot(1:length(Cm_his(2,:)), Cm_his(2,:));
%     title('Cm of layer 2');
%     figure;
%     plot(1:length(Cm_his(3,:)), Cm_his(3,:));
%     title('Cm of layer 3');
%     figure;
%     plot(1:length(Cm_his(4,:)), Cm_his(4,:));
%     title('Cm of layer 4');
%     figure;
%     plot(1:length(Cm_his(5,:)), Cm_his(5,:));
%     title('Cm of layer 5');
%     figure;
%     plot(1:length(Rm_his(1,:)), Rm_his(1,:));
%     title('Rm of layer 1');
%     figure;
%     plot(1:length(Rm_his(2,:)), Rm_his(2,:));
%     title('Rm of layer 2');
%     figure;
%     plot(1:length(Rm_his(3,:)), Rm_his(3,:));
%     title('Rm of layer 3');
%     figure;
%     plot(1:length(Rm_his(4,:)), Rm_his(4,:));
%     title('Rm of layer 4');
%     figure;
%     plot(1:length(Rm_his(5,:)), Rm_his(5,:));
%     title('Rm of layer 5');
    
%      for l = 2 : numel(cnn.layers) 
%          for j=1:outputmaps
%             spy(cnn.layers{l}.spiking{j});
%          end
%      end

    for l = 1:numel(cnn.layers)-1
       figure;
       spy(spike{l}(1:200,:));
       title(l);
    end