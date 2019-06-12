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
C = 2.1;
sigma = [5,1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001];
sigma = 0.001;
for t = 1:10
for i = 1:2000
    for j = 1:2000
         H(i,j) = train_label(i)*train_label(j)*exp(-sum(train_data(:,i)-train_data(:,j)).^2/(2*sigma(t)^2));
%         K(i,j) = exp(-sum(train_data(:,i)-train_data(:,j)).^2/(2*sigma^2));
    end
end
% Mercer_condition = eig(K);
% Positive_semi_definite = 1;
% for i = 1:2000
%     if Mercer_condition(i)<0
%         Positive_semi_definite = 0;
%     end
% end

f = -ones(2000,1);
A = [];
b = [];
Aeq = train_label';
Beq = 0;
lb = zeros(2000,1);
ub = ones(2000,1)*C;
x0 = [];
options = optimset('LargeScale','off','MaxIter',200);
Lagrange_multipliers = quadprog(H,f,A,b,Aeq,Beq,lb,ub,x0,options);

AlmostZero = (abs(Lagrange_multipliers)<max(abs(Lagrange_multipliers))/1e5);
Lagrange_multipliers(AlmostZero) = 0;
S = 0;
S = find(Lagrange_multipliers>0 & Lagrange_multipliers<C);
w = 0;
for i = S'
    w = w+Lagrange_multipliers(i)*train_label(i)*train_data(:,i);
end
for i = S'
    b0(i) = 1/train_label(i)-w'*train_data(:,i);
end
b = mean(b0);
%%
%Use the SVM to find the spam email
train_label_result = sign(train_data'*w+b);
test_label_result = sign(test_data'*w+b);
Correct_train_No = 0;
for i = 1:2000
    if (train_label(i)==1)&&(train_label_result(i)==1)||(train_label(i)==-1)&&(train_label_result(i)==-1)
        Correct_train_No = Correct_train_No+1;
    end
end
Correct_test_No = 0;
for i = 1:1536
    if (test_label(i)==1)&&(test_label_result(i)==1)||(test_label(i)==-1)&&(test_label_result(i)==-1)
        Correct_test_No = Correct_test_No+1;
    end
end

Accuracy_train(t) = Correct_train_No/2000;
Accuracy_test(t) = Correct_test_No/1536;
end






