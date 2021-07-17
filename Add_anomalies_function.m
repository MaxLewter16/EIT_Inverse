run C:\Users\lewte\OneDrive\Documents\EIDORS3D\eidors/startup.m
clear; close all;

imb = mk_common_model('a2C', 16);

%Creates image object with conductivity of 1
img = mk_image(imb.fwd_model, 1);

add_anomalies(img, 2, 6);

%Randomly add elements to a circular mesh
%num_a is number of anomalies
%num_m is number of meshes
%img is result from common model 
function img_array = add_anomalies(img, num_a, num_m) 
img_array = cell(num_m, 1);
%Sets rows and columns of subplot for viewing final meshes
if num_m > 1
        row  = round(num_m/2);
        col = round(num_m/4);
    else
        row = m;
        col = m;
end
%Creates multiple meshes
for m = 1:num_m
    clear img_2;
    img_2 = img;
    for i = 1:num_a
        [a, b] = rand_p(0,0,1); %randomly finds a point within a circle 
        circ_func = sprintf('(x-%f).^2 + (y-%f).^2<.1^2', a, b); %eq for a circle to represent anomaly
         %adds to element data of mesh using elem_select, an EIDORS
         %function
        img_2.elem_data = img_2.elem_data + 1 + elem_select(img_2.fwd_model, circ_func); 
    end
    %Add images to array 
    img_array{m} = img_2;
    %plot images
    image_m = subplot(row, col, m);
    show_fem(img_2);
end
end

%randomly finds a point within a circle
function [a, b] =rand_p(x1,y1,rc)

i=2*pi*rand;
r=sqrt(rand);
a=(rc*r)*cos(i)+x1;
b=(rc*r)*sin(i)+y1;
end


    
    
    