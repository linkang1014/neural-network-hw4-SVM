%%
%INPUT
clc;
close all;
load('train.mat');load('test.mat');
%%
%Preprocessing
%Sample Scaling
%Standardization
train_data_mean = mean(train_data');
train_data_std = std(train_data');
for i = 1:57
    for j = 1:2000
        train_data(i,j) = (train_data(i,j)-train_data_mean(i))./train_data_std(i);
    end
end
for i = 1:57
    for j = 1:1536
        test_data(i,j) = (test_data(i,j)-train_data_mean(i))./train_data_std(i);
    end
end
%%Quadratic programming
for p = 2:5
for i = 1:2000
    for j = 1:2000
%         H(i,j) = train_label(i)*train_label(j)*(train_data(:,i)'*train_data(:,j)+1)^p;
        K(i,j) = (train_data(:,i)'*train_data(:,j)+1)^p;
    end
end
% f = -ones(2000,1);
% A = [];
% b = [];
% Aeq = train_label';
% Beq = 0;
% lb = zeros(2000,1);
% C = 1e6;ub = ones(2000,1)*C;
% x0 = [];
% options = optimset('LargeScale','off','MaxIter',1000);
% Lagrange_multipliers = quadprog(H,f,A,b,Aeq,Beq,lb,ub,x0,options);
% 
% AlmostZero = (abs(Lagrange_multipliers)<max(abs(Lagrange_multipliers))/1e5);
% Lagrange_multipliers(AlmostZero) = 0;
% S = find(Lagrange_multipliers>0 & Lagrange_multipliers<C);
% w = 0;
% for i = S'
%     w = w+Lagrange_multipliers(i)*train_label(i)*train_data(:,i);
% end
% b = 1/train_label(S(1))-w'*train_data(:,S(1));
% %%
% %Use the SVM to find the spam email
% train_label_result = sign(train_data'*w+b);
% test_label_result = sign(test_data'*w+b);
% Correct_train_No = 0;
% for i = 1:2000
%     if (train_label(i)==1)&&(train_label_result(i)==1)||(train_label(i)==-1)&&(train_label_result(i)==-1)
%         Correct_train_No = Correct_train_No+1;
%     end
% end
% Correct_test_No = 0;
% for i = 1:1536
%     if (test_label(i)==1)&&(test_label_result(i)==1)||(test_label(i)==-1)&&(test_label_result(i)==-1)
%         Correct_test_No = Correct_test_No+1;
%     end
% end
% 
% Accuracy_train(p-1) = Correct_train_No/2000;
% Accuracy_test(p-1) = Correct_test_No/1536;

Mercer_condition = eig(K);
Positive_semi_definite = ones(1,4);
for i = 1:2000
    if Mercer_condition(i)<0
        Positive_semi_definite(p-1) = 0;
    end
end
end






