% run_emf.m
% Author: Junliang Guo@USTC
% Email: guojunll@mail.ustc.edu.cn

% Generalized Neural Graph Embedding with Matrix Factorization
clear;

% Options
GET_COOCCURRENCE = 0;
EXECUTE_EMF = 1;
USE_LABEL = 1;
CHECKPOINT = 0;
dataset = 'cora';
num_class = 7;
% Configuration of co-occurrence matrix
checkpoint_file = './data/save_temp/cora_100.mat';    % checkpoint file if CHECKPOINT==1
feature_file = 'cora_features.mat';                   % features of input networks
data_filename = 'cora.embeddings.walks.0';            % preprocessed random walk sequences
train_save = 'cora_train_data.npy';                   % training data saved path
vocab_filename = 'cora_dictc.txt';                    % vocabulary filename
co_occurrence_matrix_filename = 'cora_matrix.txt';    % co-occurrence matrix filename
co_occurrence_mat_outfilename = 'cora_w2v.mat';       % co-occurrence matrix filename (matlab format)
label_sample = 200;                                   % number of sampled label context
window_size = 10;                                     % window size of word2vec(toolbox) that will influence the construction of co-occurrence matrix
window_size = floor(window_size/2);                     
min_count = 0;                                        % min-count of word2vec(toolbox) that filters out words of low frequency

% Configuration of learning algorithm
maxiter = 200;                                        % maximum number of iteration of main loop 
inner_maxiter = 50;                                   % maximum number of iteration of inner loop
stepsize = 1e-7;                                      % step-size of descending/ascending
negative = 2;                                         % negative sampling parameter that is represented by k in our paper
embedding_vector_dim = 200;                           % embedding dimentionality
save_embedding_vector_filename = 'cora_vector.mat';   % filename for saving embedding vector
verbose = 5;                                          % set verbose_acc to 0, there will be no verbose description

% Run skip-gram negative sampling(SGNS) in word2vec and get the co-occurrence matrix
% where the element in i-th column and j-th row represent the co-occurrence count of i-th word and j-th word
if(GET_COOCCURRENCE)
    display('start extraction of co-occurrence matrix from SGNS');
    cd word2vec
    system('make'); % we only compile the word2vec.c
    system(['chmod u+x ', 'word2vec']);
    cmd_line = sprintf('time ./word2vec -train %s -save-vocab %s -matrix %s -output vectors.bin -saveW savedW.txt -saveC savedC.txt -nsc savednsc.txt -cbow 0 -size %d -window %d -negative %d -hs 0 -sample 1e-5 -threads 20 -binary 1 -iter 15 -min-count %d', ..., 
                        ['../data/', data_filename], ['../data/', vocab_filename], ['../data/', co_occurrence_matrix_filename], embedding_vector_dim, window_size, negative, min_count);
    display(cmd_line);
    system(cmd_line);
    cd ..
    if(USE_LABEL)
       display('add label into co-occurrence matrix');
       cmd_line = sprintf('python add_label.py %s %s %s %d %s', ['./data/', vocab_filename], ['./data/', co_occurrence_matrix_filename], ['./data/', train_save], label_sample, ['./data/', co_occurrence_matrix_filename]);
       display(cmd_line);
       system(cmd_line);
    end
    temp = load(['./data/', co_occurrence_matrix_filename]);
    w2vmatrix = spconvert(temp);
    save(['./data/', co_occurrence_mat_outfilename], 'w2vmatrix');
    display('end get co-occurrence');
end

% Run Explicit Matrix Factorization
if(EXECUTE_EMF)
    clc;
    % run EMF
    display('start EMF');
    w2vsbd(['./data/', co_occurrence_mat_outfilename], ...,
           maxiter, inner_maxiter, stepsize, negative, embedding_vector_dim, verbose, ['./data/', save_embedding_vector_filename], ...,
           ['./data/', feature_file], ['./data/', vocab_filename], ...,
           CHECKPOINT, checkpoint_file, dataset, num_class);
    display('end EMF');
end


