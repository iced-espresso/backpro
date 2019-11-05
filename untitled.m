clc;
clear;
close all;
%%
%%-------------------------------------------------------
%%--------- XOR Problem / Back Propagation -----------
%%-------------------------------------------------------
% ������ XOR Table�� �н��ϴ� NN �� �����ϴ� �����Դϴ�.
% 1-Layer, 2-Layer model�� ���� �����Ͽ� XOR ����� ���մϴ�.
% 1-Layer, 2-Layer�� model�� Back propagation�� �̿��Ͽ� �н���ŵ�ϴ�.
% �־��� ����� Ȱ���� �ֽø�, scale, ������ ����, hyper parameter���� ����� �� ������ �������� Tuning�ϼŵ� �����մϴ�.
% Layer�� Activation �Լ� Sigmoid�� �ǾƷ����� �Լ��� ����Ͻø� �˴ϴ�.
% ��� ������ ���� ó���� Initialize �� Weight,bias ���� �����Ͽ� �Բ� ÷���� �ֽñ� �ٶ��ϴ�.


%%


% Hyper parameters
% �н��� Ƚ���� Gradient update�� ���̴� step_size �Դϴ�.
% �ٸ� ���� ����Ͽ��� �����մϴ�.
epochs = 10000;
step_size = 0.1;



% Input data setting
% XOR data 
% �Է� �����͵�, XOR Table �� �°� ���ǵǾ� �ֽ��ϴ�.
train_inp = [1 1; 1 0; 0 1; 0 0];
train_out = [0; 1; 1; 0];


% Weight Setting
% �н��� ���Ǵ� Weigth,bias ���� �ʱⰪ�� ������ �ݴϴ�. �ٸ� ���� ����Ͽ��� �����մϴ�.
W1 = randn(2,3)/10;
W2 = randn(3,1)/10;
b1 = randn(1,3)/10;
b2 = randn(1,1)/10;




%-----------------------------------
%--------- Training Step -----------
%-----------------------------------
% �н��� ���۵˴ϴ�. epoch �����ŭ for ���� ���� �н��˴ϴ�.
errors = 1;

for epoch = 1:epochs

    for batch = 1:4
        % Random input ����
        patnum = randperm(4,1);
        % �Է� Xin ����
        xin = train_inp(patnum,:);
        % ���� ��� t ����
        t = train_out(patnum,1);
        
        % Layer�� �´� Forward Network ����    

        x1 = sigmoid(xin*W1 + b1);
        x2_o = x1*W2 + b2;
        x2 = sigmoid(x1*W2 + b2);
        % Binary cross entropy error�� error ǥ��
 
        error = -t*log(x2) - (1-t).*log(1-x2);
        
        %d_predicted_output = error .* (x2-t);
        
        %error = t - x2;     
       % d_predicted_output = error .* (x2 * (1-x2));
        
       % error_hidden_layer = d_predicted_output * W2.';
        %d_hidden_layer = error_hidden_layer .* (x1.*(1-x1));
        %W2 = W2 + (( x1.')* (d_predicted_output)) .* step_size;
        %b2 = b2 + sum(d_predicted_output).*step_size;
        %W1 = W1 + ((xin.') * (d_hidden_layer)).*step_size;
        %b1 = b1 + sum(d_hidden_layer).*step_size;
        % Back propagation�� ���� Weight�� Gradient update step
        
        dz2 = (x2-t);
        dW2 = (x1.')*dz2;
        db2 = dz2;
        dz1 = (dz2 * W2.') .* x1 .*(1-x1);
        dW1 = xin.' * dz1;
        db1 = dz1;
        delta_W1 = dW1;
        delta_W2 = dW2;
        delta_b1 = db1;
        delta_b2 = db2;
        
        W1 = W1 - step_size * delta_W1;
        W2 = W2 - step_size * delta_W2;
        b1 = b1 - step_size * delta_b1;
        b2 = b2 - step_size * delta_b2;
        
    end
    
    
    %epoch ������ error ����
    errors = [errors , error];
end


%% !!! weight, bias�� �����ϴ� �κ��Դϴ�. '�й�'�� �ڽ��� �й����� ��ü���ּ��� !!!
file = 'HW1_2014';
save(file,'W1','W2','b1','b2');

%-----------------------------------
%--------- Testing Step ------------
%-----------------------------------
% Test�� ���� �н��� ����� �̷�������� Ȯ��
for patnum = 1:4
    xin = train_inp(patnum,:);
     x1 = sigmoid(xin*W1 + b1);
    x2 = x1*W2 + b2;
    result = sigmoid(x2);
    if(result >0.5)
        result = 1
    else
        result = 0
    end
    
end


%-----------------------------------
%------- Activation Function -------
%-----------------------------------
function [y]= sigmoid(x)
    
  
    y = 1 ./ (exp(-x)+1);
end
