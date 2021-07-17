run C:\Users\lewte\OneDrive\Documents\EIDORS3D\eidors/startup.m
% Compare 2D algorithms
% Creates Model (model string('c2c' is 2d circ model with 576 elements), #
% of electrodes)
clear; close all;
imb = mk_common_model('c2c', 16);

%Returns the size of the vector, in this case, the number of elements
e = size(imb.fwd_model.elems, 1);
bkgnd = 1;
%Creates image object with conductivity of 1
img = mk_image(imb.fwd_model, bkgnd);

% calls forward solver 
vh = fwd_solve(img);

data = image_prior(imb.fwd_model, img);
%Add Two triangle anomalies
img.elem_data([25,37,49:50,65:66,81:83,101:103,121:124])=bkgnd * 2;
img.elem_data([95,98:100,79,80,76,63,64,60,48,45,36,33,22])=bkgnd * 2;

%implement forward solver on image with anomalies
vi = fwd_solve(img);

% Add some noise (-12db SNR, means 12 signal power o noise power)
vi_n = add_noise(2, vi, vh);
%nampl = std(vi.meas - vh.meas)*10^(-18/20);
%vi_n.meas = vi.meas + nampl *randn(size(vi.meas));

%show inhomogenous
%Pulls up image of the mesh, changes edgecolor to dark green cause why not
hh = show_fem(img); set(hh, 'EdgeColor', [.25 .5 .25])
axis square; axis off
%Print figures to a file name with an option
print_convert('tutorial120a.png', '-densiy 60')


%clf deletes/refreshes mesh but leaves things like scales and color 
%clear reset/reinitlizes variables 
clf; clear imgr igmn 

% Create inverse model
%Create eidors obj(type, name)
inv2d= eidors_obj('inv_model', 'EIT inverse');
inv2d.reconst_type='difference';
inv2d.jacobian_bkgnd.value = 1;

%Setup forward model
imb=mk_common_model('b2c', 16);
inv2d.fwd_model = imb.fwd_model;

%Guass-Newton Solver
inv2d.solve = @inv_solve_diff_GN_one_step;

%Tikhonov Prior
inv2d.hyperparameter.value= .03;
inv2d.RtR_prior = @prior_tikhonov;
imgr(1) = inv_solve(inv2d, vh, vi);
imgn(1) = inv_solve(inv2d, vh, vi_n);

% NOSER prior
inv2d.hyperparameter.value = .1;
inv2d.RtR_prior=   @prior_noser;
imgr(2)= inv_solve( inv2d, vh, vi);
imgn(2)= inv_solve( inv2d, vh, vi_n);

% Laplace image Prior
inv2d.hyperparamter.value = .1;
inv2d.RtR_prior = @prior_laplace;
imgr(3) = inv_solve(inv2d, vh, vi);
imgn(3) = inv_solve(inv2d, vh, vi_n);

% Automatic Hyperparamter selection
inv2d.hyperparamter = rmfield(inv2d.hyperparamter, 'value');
inv2d.hyperparamter.func = @choose_noise_figure;
inv2d.hyperparamter.noise_figure = .5;
invd.hyperparamter.tgt_elems = 1:4;
inv2d.RtR_prior = @prior_gaussian_HPF;
inv2d.solve = @inv_solve_diff_GN_one_step;
imgr(4) = inv_solve(inv2d, vh, vi);
imgn(4) = inv_solve(inv2d, vh, vi_n);
inv2d.hyperparamter = rmfield(inv2d.hyperparamter, 'func');


%Backprojection solver
%clear inv2d
%inv2d.solve = @inv_solve_backproj;
%inv2d.inv_solve_backproj.type= 'naive';
%imgr(5) = inv_solve(inv2d, vh, vi);
%imgn(5) = inv_solve(inv2d, vh, vi_n);
% Total variation using PDIPM
%inv2d.hyperparamter.value = 1e-5;
%inv2d.solve = @inv_solve_TV_pdipm;
%inv2d.R_prior = @prior_TV;
%inv2d.paramters.max_iterations=10;
%inv2d.paramters.term_tolerance = 1e-3;

%Vectors of structs, all structs must have the exact same fields and
%orderings
%imgr5 = inv_solve(inv2d, vh, vi);
%imgr5=rmfield(imgr5, 'type'); imgr5.type='image';
%imgr(5) = imgr5;
%imgn5= inv_solve(inv2d, vh, vi);
%imgn5 = rmfield(imgn5, 'type'); imgn5.type='image';
%imgn(5)=imgn5;

%output image
imgn(1).calc_colours.npoints = 128;
imgr(1).calc_colours.npoints = 128;
show_slices(imgr, [inf,inf,0,1,1]);
print_convert tutorial120b.png;

show_slices(imgn, [inf, inf, 0, 1, 1]);
print_convert turotial120c.png;