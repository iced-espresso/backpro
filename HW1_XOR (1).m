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

        x1 = xin*W1 + b1;
        x2 = x1*W2 + b2;
        
        % Binary cross entropy error�� error ǥ��
        error = -t*log(x2) - (1-t)*log(1-x2);
        d_predicted_output = error * sigmoid(x2);
error_hidden_layer = d_predicted_output * W2.';
d_hidden_layer = error_hidden_layer * sigmoid(x1);
        W2 += ( x1.')* (d_predicted_output) * step_size;
        b2 += sum(d_predicted_output)*lr;
        W1 += (xin.') * (d_hidden_layer)*lr;
        b1 += sum(d_hidden_layer)*lr;
        % Back propagation�� ���� Weight�� Gradient update step
        
        %delta_W1 = 
        %delta_W2 =
        %delta_b1 =
        %delta_b2 =
        
        %W1 = W1 - step_size * delta_W1;
        %W2 = W2 - step_size * delta_W2;
        %b1 = b1 - step_size * delta_b1;
        %b2 = b2 - step_size * delta_b2;
        
    end
    
    epoch
    %epoch ������ error ����
    errors = [errors , error];
end


%% !!! weight, bias�� �����ϴ� �κ��Դϴ�. '�й�'�� �ڽ��� �й����� ��ü���ּ��� !!!
file = 'HW1_�й�';
save(file,'W1','W2','b1','b2');

%-----------------------------------
%--------- Testing Step ------------
%-----------------------------------
% Test�� ���� �н��� ����� �̷�������� Ȯ��
for patnum = 1:4
    
    
end


%-----------------------------------
%------- Activation Function -------
%-----------------------------------
function [y]= sigmoid(x)
    
  
    y = 1 ./ (exp(-x)+1);
end

  
