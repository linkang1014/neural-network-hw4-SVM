%%
%INPUT
clc;
close all;
load('train.mat');load('test.mat');
eval_total_data = [train_data,test_data];
eval_total_label = [train_label;test_label];
r = randperm(size(eval_total_data,2));
eval_data = eval_total_data(:,r);
eval_data = eval_data(:,1:700);

eval_label = eval_total_label(r,:);
eval_label = eval_label(1:700,:);
save('eval.mat','eval_data','eval_label');

