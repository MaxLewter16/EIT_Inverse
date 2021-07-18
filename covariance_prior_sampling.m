clear 
close all

%decide model type, # of electrodes, number of meshes, and arrangement of
%subplot
mdl_type = 'd2C';
n_electrodes = 8;
n_trials = 4;
subplot_dim = [2 2];

%make model w/ 8 electrodes & CEM
circle = mk_common_model(mdl_type,n_electrodes);
%get number of elements for dynamic function
n_elem = length(circle.fwd_model.elems);
%make background conductivity
img_1 = mk_image(circle,1); 

%getting coords of each elem's center to find distance btwn them
coordinates =[];
for element = 1:n_elem
    nodes = img_1.fwd_model.elems(element,:); %get nodes from corners of each elem
    x = []; y = [];
    for z = 1:3 %3 bc each elem has 3 corners 
        %gets node coordinates for each elem
        x(1,z) = img_1.fwd_model.nodes(nodes(1,z),1);
        y(1,z) = img_1.fwd_model.nodes(nodes(1,z),2);
    end
    %avg node coordinates
    x_avg = sum(x)/3;
    y_avg = sum(y)/3;
    %puts all coords into an array
    coordinates(element,1) = x_avg;
    coordinates(element,2) = y_avg;
    
end

a = .5; 
b = .2; %smoothness parameter
covar = [];
for i = 1:n_elem
    for j = 1:n_elem
        dist = (coordinates(i,1) - coordinates(j,1))^2 + (coordinates(i,2) - coordinates(j,2))^2;
        if i == j
            c = 1; %a + c = variance
        else
            c = 0;
        end
        covar(i,j) = a*exp(-(dist)/(2*b^2)) + c;
    end
end


samples = mvnrnd(ones(n_elem,1),covar,4); %(mu, covariance matrix, number of samples)
%samples = samples'; %transposing so that each column is one set of samples
%Now uses this solution to compute the POD with the wrapper:
[U_POD, S_POD, V_POD] = pod(samples);
%%
%Plots the mode energies
ModeEnergies=S_POD.^2;
ModeEnergyFraction=ModeEnergies/sum(ModeEnergies);
figure('Color','w','Position',[146 620 403 357]);
bar(1:length(ModeEnergies),ModeEnergyFraction,'k');
title('Mode Energies');
%%{
circle = mk_common_model(mdl_type,n_electrodes);
for i = 1:n_trials
    img_1t = mk_image(circle,samples(:,i));
    subplot(subplot_dim(1),subplot_dim(2),i);
    show_fem(img_1t);
end
%%}