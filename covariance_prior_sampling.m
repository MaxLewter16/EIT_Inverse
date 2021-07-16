clear 
close all

circle = mk_common_model('a2C',8); 
img_1 = mk_image(circle,1); 

coordinates =[];
for element = 1:64
    nodes = img_1.fwd_model.elems(element,:);
    x = [];
    y = [];
    for z = 1:3
        x(1,z) = img_1.fwd_model.nodes(nodes(1,z),1);
        y(1,z) = img_1.fwd_model.nodes(nodes(1,z),2);
    end
    x_avg = sum(x)/3;
    y_avg = sum(y)/3;
    coordinates(element,1) = x_avg;
    coordinates(element,2) = y_avg;
    
end

a = .5;
b = .2; %smoothness parameter
covar = [];
for i = 1:64
    for j = 1:64
        dist = (coordinates(i,1) - coordinates(j,1))^2 + (coordinates(i,2) - coordinates(j,2))^2;
        if i == j
            c = 1; %a + c = variance
        else
            c = 0;
        end
        covar(i,j) = a*exp(-(dist)/(2*b^2)) + c;
    end
end

samples = mvnrnd(ones(64,1),covar,4); %(mu, covariance matrix, number of samples)
samples = samples'; 

%{
circle = mk_common_model('a2C',8);
for i = 1:4
    img_1t = mk_image(circle,r(:,i));
    subplot(2,2,i);
    show_fem(img_1t);
end
%}