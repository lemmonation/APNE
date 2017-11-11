% w2vsbd.m
% EMF block minimization/maximization
% Author: Junliang Guo@USTC
% Email: guojunll@mail.ustc.edu.cn

% Objective:  min MF(D, C*W)
% Algorithm:
%   While 1
%       while
%          C = C - MFgrad(C);
%       while
%          W = W - MFgrad(D);
%   Until Converge

function w2vsbd(co_mat_filename, maxiter, ...,
                inner_maxiter, stepsize, k, dim, verbose_acc, save_embedding_vector_filename, ...,
                feature_file,  dict_file, ...,
                CHECKPOINT, checkpoint_file, dataset, class_num)
    load(co_mat_filename);
    rand('state',2014);
     
    H = full(w2vmatrix);
    disp(['size: ', num2str(size(H))]);
    
    display('process the feature matrix');
    t_file = 'data/temp.mat';
    cmd_line = sprintf('python p_feature.py %s %s %s', feature_file, t_file, dict_file);
    %display(cmd_line);
    system(cmd_line);
    display('process end')
    load(t_file);

    F = full(features);
    F = F(1:size(w2vmatrix, 1), :);
    d_f = size(F, 2);

    randlist = [];
    [sample_num, context_num] = size(H);
    D = H';
    eff = 1/k;

    % construct Q
    Qw = sum(H, 2);
    Qc = sum(H, 1);
    Qnum = sum(sum(H));
    Qtemp = Qw*Qc./(eff*Qnum);
    Q = Qtemp + H;

    if (CHECKPOINT)
      load(checkpoint_file);
      W = W_t
      S = S_t
    else
      % random initialize
      W = (rand(dim, sample_num) - 0.5)/dim/200;
      S = (rand(d_f, dim) - 0.5) / dim;
    end
    % Use GPU acceleration
    S = gpuArray(S);
    D = gpuArray(D);
    Q = gpuArray(Q);
    W = gpuArray(W);
    F = gpuArray(F);

    accuracy_list = [];
    err_list = [];

    for iter = 1:maxiter
        W_last = W;
        S_last = S;
        
        if mod(iter,2)
            % descent W
            for inner_iter = 1:inner_maxiter
                ED = Q'.*(1./(1 + exp(-F*S*W)));
                recons = D - ED;
                W_grad = S'*F'*recons;
                W = W + stepsize*W_grad;
            end
        else
            % descent S
            for inner_iter = 1:inner_maxiter
                ED = Q'.*(1./(1 + exp(-F*S*W)));
                recons = D - ED;
                S_grad = F'*recons*W';
                S = S + stepsize*S_grad;
            end
        end

        if 0 == mod(iter, verbose_acc)
            err = norm(recons, 'fro');
            err_list = [err_list err];
            W_reg_fro = norm(W, 'fro')/sample_num;
            S_reg_fro = norm(S, 'fro')/sample_num;
            disp(['epoch:', num2str(iter),',err:', num2str(err), ',W:', num2str(W_reg_fro), ...,
              ',S:', num2str(S_reg_fro), ',stepsize:', num2str(stepsize)]);
            % Save checkpoint and test
            save_temp = ['data/save_temp/', dataset, num2str(iter), '.mat'];
            W_t = gather(W);
            S_t = gather(S);
            save(save_temp, 'W_t', 'S_t');
            cmd_line = sprintf('python test.py %s %s %s %d', save_temp, dict_file, dataset, class_num);
            system(cmd_line);
        end
    end
    W = gather(W);
    S = gather(S);
    save(save_embedding_vector_filename, 'W', 'S');
end
