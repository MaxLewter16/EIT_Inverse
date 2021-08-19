close all
clear

circle = mk_common_model('a2C',8);
img_1 = mk_image(circle,1);

[stim, meas] = mk_stim_patterns(8,1,'{ad}','{ad}',{'meas_current'},1);
img_1.fwd_model.stimulation = stim;
img_1.fwd_model.meas_select = meas;
J = calc_jacobian(img_1); %calculate jacobian from homogenous background image
homog_volt = fwd_solve(img_1).meas;

select_fcn = '(x-.1).^2+(y-.2).^2<0.3^2'; %creating anomaly
img_1.elem_data =1 + elem_select(img_1.fwd_model, select_fcn);
select_fcn2 = '(x-.1).^2+(y-.2).^2< 0.6^2'; 
img_1.elem_data =img_1.elem_data + .5*elem_select(img_1.fwd_model, select_fcn2);
 
 
inhomog = fwd_solve(img_1);
noisy_data = add_noise(70,inhomog); %add noise to boundary voltage measurements 
volt2 = noisy_data.meas;


y_values = volt2; %voltage measurement data

%making covariance matrix
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
b = .2;
test = 0;
covar = [];
for i = 1:64
    for j = 1:64
        dist = (coordinates(i,1) - coordinates(j,1))^2 + (coordinates(i,2) - coordinates(j,2))^2;
        if i == j
            c = .00001;
        else
            c = 0;
        end
        covar(i,j) = a*exp(-(dist)/(2*b^2)) + c;
    end
end


CondPriorMean = 1;
% CondPriorSigma = 1; use covariance matrix instead
LogNoiseVarianceMean = 0;
LogNoiseVarianceSigma = 1;


logpdf = @(Parameters)logPosterior(Parameters,y_values, ...
    CondPriorMean,covar, ...
    LogNoiseVarianceMean,LogNoiseVarianceSigma, J, homog_volt);

Cond = 1 + randn(64,1); %starting values around background conductivity of 1
LogNoiseVariance = randn; %starting values around 0

startpoint = [Cond; LogNoiseVariance];
smp = hmcSampler(logpdf,startpoint,'UseNumericalGradient',true);

[MAPpars,fitInfo] = estimateMAP(smp,'VerbosityLevel',0);
MAPCond = MAPpars(1:end-1)
MAPLogNoiseVariance = MAPpars(end)
%MAPpars(end) = -0.01;

%create image of MAP conductivity estimate
circle = mk_common_model('a2C',8);
img_1t = mk_image(circle,MAPCond);
show_fem(img_1t);
eidors_colourbar(img_1t);


[smp,tuneinfo] = tuneSampler(smp,'Start',MAPpars);

NumChains = 2;
chains = cell(NumChains,1);
Burnin = 500;
NumSamples = 5000;
for c = 1:NumChains
    if (c == 1)
        level = 1;
    else
        level = 0;
    end
    chains{c} = drawSamples(smp,'Start',MAPpars + .2*randn(size(MAPpars)), ...
        'Burnin',Burnin,'NumSamples',NumSamples, ...
        'VerbosityLevel',level,'NumPrint',500);
end

diags = diagnostics(smp,chains)


function logpdf = logPosterior(Parameters,Y, ...
    CondPriorMean,covar, ...
    LogNoiseVarianceMean,LogNoiseVarianceSigma, jacobian, homog_volt)

% Unpack the parameter vector
Cond        = Parameters(1:end-1);
LogNoiseVariance = Parameters(end);
Sigma                   = sqrt(exp(LogNoiseVariance)); %convert from log
Variance = Sigma^2;
v_covar = Variance*eye(64); %voltage covarance matrix is diagonal matrix of variance
Mu                      = fwdsolver(Cond, jacobian, homog_volt);
Z                       = (Y - Mu)/Sigma;

loglik1 = -(64/2)*log(2*pi) - .5*log(det(v_covar)) - .5*Z'*Z;

% Compute log priors
LPCond           = multi_normalPrior(Cond,CondPriorMean,covar);
LPLogNoiseVar  = normalPrior(LogNoiseVariance,LogNoiseVarianceMean,LogNoiseVarianceSigma);
logprior                                = LPCond + LPLogNoiseVar;
% Return the log posterior
logpdf  = loglik1 + logprior;
end

function logpdf = multi_normalPrior(P,Mu,covar)
Z          = (P - Mu);
logpdf     = (-.5*log(det(covar)) - (64/2)*log(2*pi) - .5*Z'*inv(covar)*Z);
end

function logpdf = normalPrior(P,Mu,Sigma)
Z          = (P - Mu)./Sigma;
logpdf     = (-log(Sigma) - .5*log(2*pi) - sum(.5*(Z.^2)));
end

function voltages = fwdsolver(conductivity, jacobian, homog_volt)

voltages = homog_volt + jacobian*(conductivity - ones(64,1));

end

function diff_voltages = diff_fwdsolver(diff_conductivity, jacobian )

diff_voltages = jacobian*(diff_conductivity);

end

