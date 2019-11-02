%% Constants
N_input_char    = 15; % number of training characters
char_matrix_dim = 7; % character square matrix edge
neur_output     = 2; % # of neurons in the output layer
neur_hidden     = 8; % # of neurons in the hidden layer
neur_input      = 3; % # of neurons in the input layer
nu              = 0.4; 
max_error = N_input_char*2*0.25; % maximum error allowed is 25%
%% Training matrix initialization and representation
% 15 characters, 5 zeros - 5 C - 5 F.
% 0
x1 = [1 1 1 1 1 1 1;1 -1 -1 -1 -1 -1 1;1 -1 -1 -1 -1 -1 1;
    1 -1 -1 -1 -1 -1 1;1 -1 -1 -1 -1 -1 1;1 -1 -1 -1 -1 -1 1;1 1 1 1 1 1 1];
x2 = [-1 1 1 1 1 1 1; 1 -1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 -1 1;
    1 -1 -1 -1 -1 -1 1;1 -1 -1 -1 -1 -1 1;1 -1 -1 -1 -1 -1 1;1 1 1 1 1 1 1];
x3 = [1 1 1 1 1 1 1; 1 -1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 -1 1;
    1 -1 -1 -1 -1 -1 1;1 -1 -1 -1 -1 -1 1;1 1 -1 -1 -1 -1 1;-1 1 1 1 1 1 1];
x4 = [1 1 1 1 1 1 1;1 -1 -1 -1 -1 1 1;1 -1 -1 -1 -1 -1 1;
    1 -1 -1 -1 -1 -1 1;1 -1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 -1 1; 1 1 1 1 1 1 1];
x5 = [1 1 1 1 1 1 -1;1 -1 -1 -1 -1 -1 1;1 -1 -1 -1 -1 -1 1;
    1 -1 -1 -1 -1 -1 1;1 -1 -1 -1 -1 -1 1;1 -1 -1 -1 -1 -1 1;-1 1 1 1 1 1 1];
% C
x6 = [-1 1 1 1 1 1 1;1 -1 -1 -1 -1 -1 -1;1 -1 -1 -1 -1 -1 -1;
    1 -1 -1 -1 -1 -1 -1;1 -1 -1 -1 -1 -1 -1;1 -1 -1 -1 -1 -1 -1;-1 1 1 1 1 1 1];
x7 = [1 1 1 1 1 1 1; 1 -1 -1 -1 -1 -1 -1;1 -1 -1 -1 -1 -1 -1;1 -1 -1 -1 -1 -1 -1;
    1 -1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1 -1; 1 1 1 1 1 1 1];
x8 = [1 1 1 1 1 1 -1;1 -1 -1 -1 -1 -1 -1;1 -1 -1 -1 -1 -1 -1;1 -1 -1 -1 -1 -1 -1;
    1 -1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1 -1; 1 1 1 1 1 1 1];
x9 = [1 1 1 1 1 1 1;1 -1 -1 -1 -1 -1 -1;1 -1 -1 -1 -1 -1 -1;1 -1 -1 -1 -1 -1 -1;
    1 -1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1 1; 1 1 1 1 1 1 1];
x10 = [1 1 1 1 1 1 -1;1 -1 -1 -1 -1 -1 -1;1 -1 -1 -1 -1 -1 -1;1 -1 -1 -1 -1 -1 -1;
    1 -1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1 -1; 1 1 1 1 1 1 -1];
% F
x11 = [1 1 1 1 1 1 1;1 1 -1 -1 -1 -1 -1;1 1 1 1 1 1 1;1 -1 -1 -1 -1 -1 -1;
    1 -1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1 -1];
x12 = [1 1 1 1 1 1 1;1 -1 -1 -1 -1 -1 -1;1 1 1 1 1 1 1; 1 -1 -1 -1 -1 -1 -1;
    1 -1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1 -1; -1 -1 -1 -1 -1 -1 -1];
x13 = [1 1 1 1 1 1 -1;1 -1 -1 -1 -1 -1 -1;1 1 1 1 1 1 -1;1 -1 -1 -1 -1 -1 -1;
    1 -1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1 -1];
x14 = [1 1 1 1 1 1 1;1 -1 -1 -1 -1 -1 -1;1 1 1 1 1 1 1;1 -1 -1 -1 -1 -1 -1;
    1 -1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1 -1];
x15 = [1 1 1 1 1 1 1;1 -1 -1 -1 -1 -1 -1;1 1 1 1 1 -1 -1;1 -1 -1 -1 -1 -1 -1;
    1 -1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1 -1];
xr1 = reshape(x1',1,49); xr2 = reshape(x2',1,49); xr3 = reshape(x3',1,49);
xr4 = reshape(x4',1,49); xr5 = reshape(x5',1,49); xr6 = reshape(x6',1,49);
xr7 = reshape(x7',1,49); xr8 = reshape(x8',1,49); xr9 = reshape(x9',1,49);
xr10 = reshape(x10',1,49); xr11 = reshape(x11',1,49); xr12 = reshape(x12',1,49);
xr13 = reshape(x13',1,49); xr14 = reshape(x14',1,49); xr15 = reshape(x15',1,49);
% matrix containing in each column a full character
X = [xr1' xr2' xr3' xr4' xr5' xr6' xr7' xr8' xr9' xr10' xr11' xr12' xr13' xr14' xr15'];
figure % display all the training characters
for i=1:N_input_char
    subplot(3,5,i);
    imshow((reshape(X(:,i),char_matrix_dim,char_matrix_dim))','initialMagnification','fit')
end
%% Initialization
% Assign random weights in the range {-1,1}
w_input = (rand(neur_input,char_matrix_dim^2)-0.5)*2;
w_hidden = (rand(neur_hidden,neur_input)-0.5)*2;
w_output = (rand(neur_output,neur_hidden)-0.5)*2;
disp 'Initial random weights'
w_input;
w_hidden;
w_output;
% Given the weights evaluate z and y parameters for the three layers. Also
% count the number of errors with respect the expected output.
[z_input,w_input,y_input,z_hidden,w_hidden,y_hidden,z_output,w_output,y_output,N_errors] = evaluation(w_input,w_hidden,w_output,X,neur_hidden,neur_input);
disp 'Initial errors with random weights'
N_errors;
eff = (1-N_errors/30)*100;
% Create a matrix to keep track of the weights adjustments
max_neur_per_layer = max(neur_hidden,neur_input);
index = zeros(3,max_neur_per_layer);
%%%% Weights adjustment of single neurons starting from the outer layer
%% OUTPUT LAYER - Weight adjustment
% Applying minimum disturbance principle. Sort weight in increasing fashion
output_ind = zeros(1,neur_output);
z_out_asc = sort(abs(z_output));
for i = 1:neur_output
    for k = 1:neur_output
        if z_out_asc(i) == abs(z_output(k))
            output_ind(i) = k;
        end
    end
end
% Change weights and keep the change only if it leads to better performance
for i = 1:neur_output
	if (N_errors>max_error)
    [w_min]=adjust_weights(z_output,w_output,N_errors,y_hidden,nu,output_ind(i),neur_hidden);
      w_output_min = w_min;
        
    [z_input,w_input,y_input,z_hidden,w_hidden,y_hidden,z_output_min,w_output_min,y_output_min,N_errors_tmp]=evaluation(w_input,w_hidden,w_output_min,X,neur_hidden,neur_input);
    
    if (N_errors_tmp<N_errors)
      N_errors = N_errors_tmp;
      w_output = w_output_min;
      z_output = z_output_min;
      y_output = y_output_min;
      index(1,output_ind(i)) = 1;
    end
  end
end
disp 'Output layer after weights adjustment'
w_output;
disp 'Error'
N_errors;
eff = (1-N_errors/30)*100;
%% HIDDEN LAYER - Weight adjustment
% Applying minimum disturbance principle. Sort weight in increasing fashion
hidden_ind = zeros(1,neur_hidden);
z_hid_asc = sort(abs(z_hidden));
for i = 1:neur_hidden
    for k = 1:neur_hidden
        if z_hid_asc(i) == abs(z_hidden(k))
            hidden_ind(i) = k;
        end
    end
end
% Change weights and keep the change only if it leads to better performance
for i = 1:neur_hidden
  if ((N_errors>max_error))
    [w_min]=adjust_weights(z_hidden,w_hidden,N_errors,y_input,nu,hidden_ind(i),neur_input);
    w_hidden_min = w_min;
    [z_input, w_input, y_input, z_hidden_min, w_hidden_min, y_hidden_min, z_output_min, w_output,y_output_min,N_errors_tmp]=evaluation(w_input,w_hidden_min,w_output,X,neur_hidden,neur_input);
    if (N_errors_tmp<N_errors)
        N_errors = N_errors_tmp;
        w_hidden = w_hidden_min;
        y_hidden = y_hidden_min;
        z_hidden = z_hidden_min;
        z_output = z_output_min;
        y_output = y_output_min;
        index(2,hidden_ind(i)) = 1;
    end
  end
end
disp 'Hidden layer after weights adjustment'
w_hidden;
disp 'Error'
N_errors;
eff = (1-N_errors/30)*100;
%% INPUT LAYER - Weight adjustment
% Applying minimum disturbance principle. Sort weight in increasing fashion
input_ind = zeros(1,neur_input);
z_hid_asc = sort(abs(z_input));
for i = 1:neur_input
    for k = 1:neur_input
        if z_hid_asc(i) == abs(z_input(k))
            input_ind(i) = k;
        end
    end
end
% Change weights and keep the change only if it leads to better performance
for i = 1:neur_input
  if (N_errors>max_error)
    [w_min]=adjust_weights(z_input,w_input,N_errors,X(:,11),nu,input_ind(i),char_matrix_dim^2);
    w_input_min = w_min;
    [z_input_min,w_input_min,y_input_min,z_hidden_min,w_hidden,y_hidden_min,z_output_min,w_output,y_output_min,N_errors_tmp]=evaluation(w_input_min,w_hidden,w_output,X,neur_hidden,neur_input);
    if (N_errors_tmp<N_errors)
        N_errors = N_errors_tmp;
        w_input = w_input_min;
        z_input = z_input_min;
        y_input = y_input_min;
        z_hidden = z_hidden_min;
        y_hidden = y_hidden_min;
        z_output = z_output_min;
        y_output = y_output_min;
        index(3,input_ind(i)) = 1;
    end
  end
end
disp 'Input layer after weights adjustment'
w_input;
disp 'Error'
N_errors;
eff = (1-N_errors/30)*100;
%%%% Weights adjustment of neuron pairs starting from the outer layer
%% Combined OUTPUT Neurons - Weight change
% Change weights and keep the change only if it leads to better performance
if ((index(3,[1:neur_output])~=1)&(N_errors>max_error))
  w_output_two = w_output;
  for j = 1:neur_hidden
    w_output_two([1:2],j) = w_output([1:2],j)+2*nu*y_hidden(j)*N_errors;
  end
  [z_input,w_input,y_input,z_hidden,w_hidden,y_hidden,z_output_min,w_output_two,y_output_min,N_errors_tmp]=evaluation(w_input,w_hidden,w_output_two,X,neur_hidden,neur_input);
  if (N_errors_tmp < N_errors)
    N_errors = N_errors_tmp;
    z_output = z_output_min;
    y_output = y_output_min;
    w_output = w_output_two;
    
    index(1,output_ind(1)) = 1;
    index(1,output_ind(2)) = 1;
  end
end
disp 'PAIRS: Output layer after weights adjustment'
w_output;
disp 'Error'
N_errors;
eff = (1-N_errors/30)*100;
%% Combined HIDDEN Neurons - Weight change
% Change weights and keep the change only if it leads to better performance
for i=1:neur_hidden-1
  if (index(2,hidden_ind(i))~=1) & (N_errors>max_error)
    for j=i:neur_hidden
      if (index(2,hidden_ind(j))~=1)
        w_hidden_two = w_hidden;
        k = hidden_ind(i);
        l = hidden_ind(j);
        for m = 1 : neur_input
            w_hidden_two(k,m) = w_hidden_two(k,m) + 2*nu*y_input(m)*N_errors;
            w_hidden_two(l,m) = w_hidden_two(l,m) + 2*nu*y_input(m)*N_errors;
        end
      [z_input,w_input,y_input,z_hidden_min,w_hidden_two,y_hidden_min,z_output_min,w_output,y_output_min,N_errors_tmp]=evaluation(w_input,w_hidden_two,w_output,X,neur_hidden,neur_input);
        if (N_errors_tmp < N_errors),
            N_errors = N_errors_tmp;
            w_hidden = w_hidden_two;
            y_hidden = y_hidden_min;
            z_hidden = z_hidden_min;
            z_output = z_output_min;
            y_output = y_output_min;
            index(2,hidden_ind(i))=1;
            index(2,hidden_ind(j))=1;
        end
      end
    end
  end
end
disp 'PAIRS: Hidden layer after weights adjustment'
w_hidden;
disp 'Error'
N_errors;
eff = (1-N_errors/30)*100;
%% Combined INPUT Neurons - Weight change
% Change weights and keep the change only if it leads to better performance
for i=1:neur_input-1
  if (index(1,input_ind(i))~=1) & (N_errors>max_error)
    for j=i:neur_input
      if (index(1,input_ind(j))~=1)
        w_input_two = w_input;
        k = input_ind(i);
        l = input_ind(j);
        for m = 1 : char_matrix_dim^2
            w_input_two(k,m) = w_input_two(k,m) + 2*nu*X(m,11)*N_errors;
            w_input_two(l,m) = w_input_two(l,m) + 2*nu*X(m,11)*N_errors;
        end
        
        [z_input_min, w_input_two, y_input_min, z_hidden_min, w_hidden,y_hidden_min, z_output_min, w_output,y_output_min, N_errors_tmp] =evaluation(w_input_two, w_hidden, w_output, X, neur_hidden, neur_input);
        if (N_errors_tmp<N_errors)
          N_errors = N_errors_tmp;
          w_input = w_input_two;
          y_input = y_input_min;
          z_input = z_input_min;
          y_hidden = y_hidden_min;
          z_hidden = z_hidden_min;
          z_output = z_output_min;
          y_output = y_output_min;
          index(1,input_ind(i))=1;
          index(1,input_ind(j))=1;
        end
      end
    end
  end
end
disp 'PAIRS: Input layer after weights adjustment'
w_input;
disp 'Error'
N_errors;
eff = (1-N_errors/30)*100;
%% Combined triplet HIDDEN Neurons - Weight change
% Change weights and keep the change only if it leads to better performance
for i=1:neur_hidden-2
  if (index(2,hidden_ind(i))~=1) & (N_errors>max_error)
    for j=i:neur_hidden-1
      if (index(2,hidden_ind(j))~=1)
        for k=j:neur_hidden
          if (index(2,hidden_ind(k))~=1)
            w_hidden_two = w_hidden;
            for l = 1 : neur_input
w_hidden_two(hidden_ind(i),l)=w_hidden_two(hidden_ind(i),l)+2*nu*y_input(l)*N_errors;
w_hidden_two(hidden_ind(j),l)=w_hidden_two(hidden_ind(j),l)+2*nu*y_input(l)*N_errors;
w_hidden_two(hidden_ind(k),l)=w_hidden_two(hidden_ind(k),l)+2*nu*y_input(l)*N_errors;
            end
[z_input,w_input,y_input,z_hidden_min,w_hidden_two,y_hidden_min,z_output_min,w_output,y_output_min,N_errors_tmp]=evaluation(w_input,w_hidden_two,w_output,X,neur_hidden,neur_input);
            if (N_errors_tmp<N_errors)
                N_errors = N_errors_tmp;
                w_hidden = w_hidden_two;
                y_hidden = y_hidden_min;
                z_hidden = z_hidden_min;
                z_output = z_output_min;
                y_output = y_output_min;
                index(2,hidden_ind(i))=1;
                index(2,hidden_ind(j))=1;
            end
          end
        end
      end
    end
  end
end
disp 'TRIPLETS: Hidden layer after weights adjustment. FINAL CONFIGURATION'
w_hidden;
disp 'Error'
N_errors;
disp 'Training efficiency'
eff = (1-N_errors/30)*100;