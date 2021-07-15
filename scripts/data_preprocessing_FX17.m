%% prepare environment
clear all
addpath(genpath('X:\Lab2\'))
addpath(genpath('X:\Stack')) % Add path of main folder
addpath('X:\')

%% Relative reflectance conversion
FOLD ='X:\Stack\Lab';                                                      % Put folder name of level 0 data here
visIm = '1-7_FX10_0000';                                                   % Name of FX10 image (VIS-NIR)
irIm =  '1-7_FX17_0000';                                                   % Name of FX17 image (NIR-SWIR)

%First seperate out the dark reference
fullname = [FOLD filesep visIm '\capture\DARKREF_' visIm];
[im, info] = enviread([fullname '.raw'],[fullname '.hdr']);
wavelengthsImec = cellfun(@str2num,strsplit(info.Wavelength(2:end-1),','));
Dark=im;

%Then seperate out the white reference
fullname = [FOLD filesep visIm '\capture\WHITEREF_' visIm];
[im, info] = enviread([fullname '.raw'],[fullname '.hdr']);
wavelengthsImec = cellfun(@str2num,strsplit(info.Wavelength(2:end-1),','));
Hal=im;

%Now we can calculate the mean White and mean dark spectrum at every pixel
%across the image.
da=mean(Dark);
ha=mean(Hal(:,:,:));                                                       % Select specific bands, should not all bands be used if errors are present
clear Dark Hal Darkvis Halvis im

%Now LOAD THE Image to correct
fullname = [FOLD filesep visIm '\capture\' visIm]; 
[im, info] = enviread([fullname '.raw'],[fullname '.hdr']);
wavelengthsImec = cellfun(@str2num,strsplit(info.Wavelength(2:end-1),','));

%Correction
HimC2=zeros(size(im));
for i=1:size(im,1)
    Correct_visIm(i,:,:)=((im(i,:,:)-da)./(ha-da));
end

%Resave it for further analysis
enviwrite(Correct_visIm,info,[FOLD filesep visIm filesep visIm 'Corrected.raw'],[FOLD filesep visIm filesep visIm 'Corrected.hdr']);

%% Calculate intensity normalised values
fullname = [FOLD filesep irIm '\' irIm 'Corrected']; 
[Correct_irIm, info] = enviread2([fullname '.raw'],[fullname '.hdr']);     % Load the corrected NIR-SWIR image

%intensity normalisation FX17
intensity_norm_ir = zeros(size(Correct_irIm));
ir_dims = zeros(size(Correct_irIm));

for i=1:size(ir_dims,1)
    for j=1:size(ir_dims,2)
        min_irval = min(Correct_irIm(i,j,:));
        max_irval = max(Correct_irIm(i,j,:));
        intensity_norm_ir(i,j,:) = (Correct_irIm(i,j,:)-min_irval)./(max_irval-min_irval);
    end
end