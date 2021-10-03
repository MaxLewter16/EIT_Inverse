function [jacobian,homog_voltages, voltages] = calc_j_hvoltages()
%calc_j_hvoltages calculates the jacobian and the homogenous voltages
circle = mk_common_model('a2C',8);
img = mk_image(circle,1);

[stim, meas] = mk_stim_patterns(8,1,'{ad}','{ad}',{'meas_current'},1);
img.fwd_model.stimulation = stim;
img.fwd_model.meas_select = meas;
jacobian = calc_jacobian(img); %calculate jacobian from homogenous background image
homog_voltages = fwd_solve(img).meas;



img.elem_data = 1 + elem_select(img.fwd_model,'[(x+.1).^2 + (y+.1).^2<.2^2]'); 

show_fem(img);
eidors_colourbar(img);
 
inhomog = fwd_solve(img);
noisy_data = add_noise(70,inhomog); %add noise to boundary voltage measurements 
voltages = noisy_data.meas;
end



