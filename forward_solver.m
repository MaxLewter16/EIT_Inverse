
function voltages = fwdsolver(conductivity)

run C:/Users/krait/Documents/EIDORS/eidors-v3.10/eidors/startup.m %change to correct startup line
circle= mk_common_model('a2d1c',4);

conductivity = transpose(conductivity); %oriented incorrectly in data file upload
img_1 = mk_image(circle,conductivity); %assign conductivity values to image

[stim, meas] = mk_stim_patterns(4,1,'{ad}','{ad}',{'meas_current'},10);
img_1.fwd_model.stimulation = stim;
img_1.fwd_model.meas_select = meas;

voltages = fwd_solve(img_1);

end

