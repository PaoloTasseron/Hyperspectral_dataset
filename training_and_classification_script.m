%% prepare environment
clear; close all; clc                                                      % clear environment and close all
addpath(genpath('X:\Lab2\'))                                               % Add path of main folder
addpath(genpath('X:\Stack'))                                               % Add path of main folder

%% LOAD SNAPSCAN IMAGES
FOLD = 'E:\2021-05-28_Plastic_ SWIR'                                       % Name of the image folder
irIm = 'f5.6_25ms_01_corrected'                                            % Name of the image

fullname = [FOLD filesep irIm];                                            % Specify image path
[im, info] = enviread([fullname '.raw'],[fullname '.hdr']);                % Load the corrected SNAPSCAN image

sdimage(im)                                                                % Show SD image 
% annotate pixels of each class  
% 'create dataset in workspace' -> training_data_outdoor.mat
% 'create current labels in workspace' -> training_data_outdoor_class.mat

%% Loading in data and removing 'unknown' class (only required after first annotating the pixels)
%file_data = load('E:\training_dataset_north.mat');                        % w35ms_f5.6_60ms_north_03_corrected
file_data = load('X:\Riverbank-litter\training_data_outdoor.mat');         % f5.6_25ms_01_corrected

fns = fieldnames(file_data);                                               % remove 'unknown' class for classifying
file_data = file_data.(fns{1});                                            % remove 'unknown' class for classifying
dim = size(file_data,3);                                                   % remove 'unknown' class for classifying
file_data = file_data(:,:,2:dim);                                          % remove 'unknown' class for classifying
dataset_outdoor = file_data;                                               % create new dataset name

% save new training dataset under these names:
% training_dataset_north.mat    (annotated pixels based on w35ms_f5.6_60ms_north_03_corrected)
% dataset_outdoor.mat           (annotated pixels based on f5.6_25ms_01_corrected)

%% Creating training data and relabeling the annotated ROIs (only required once)
load('X:\final_datasets\dataset_outdoor.mat');                             %includes: water, vegetation, wood, rock, plastic
load('X:\final_datasets\training_dataset_north.mat');                      %includes: water, vegetation, wood, plastic, sand

training_dataset_relabelled  =      sdrelab(training_dataset_north,{'water','water','vegetation','vegetation','wood','wood','rock','rock','sample_1','plastic','sample_2','plastic','sample_3','plastic','sample_4','plastic','sample_5','plastic','sample_6','plastic','sample_7','plastic','sample_8','plastic','sample_9','plastic','sample_10','plastic','sample_11','plastic','sample_12','plastic','sample_13','plastic','sample_13','plastic','sample_14','plastic','sample_15','plastic','sample_16','plastic','sample_17','plastic','sample_18','plastic','sample_19','plastic','sample_20','plastic','sample_21','plastic','sample_22','plastic','sample_23','plastic','sample_24','plastic','sample_25','plastic','sample_26','plastic'});
dataset_outdoor_relabelled   =      sdrelab(dataset_outdoor,{'water','water','vegetation','vegetation','wood','wood','sand','sand','sample_1','plastic','sample_2','plastic','sample_3','plastic','sample_4','plastic','sample_5','plastic','sample_6','plastic','sample_7','plastic','sample_8','plastic','sample_9','plastic','sample_10','plastic','sample_11','plastic','sample_12','plastic','sample_13','plastic','sample_13','plastic','sample_14','plastic','sample_15','plastic','sample_16','plastic','sample_17','plastic','sample_18','plastic','sample_19','plastic','sample_20','plastic','sample_21','plastic','sample_22','plastic','sample_23','plastic','sample_24','plastic','sample_25','plastic','sample_26','plastic'});
dataset_all_pixels           =      [training_dataset_relabelled;dataset_outdoor_relabelled];

% training_dataset_relabelled saved as 'training_dataset_relabelled.mat'
% dataset_outdoor_relabelled saved as 'dataset_outdoor_relabelled.mat'
% dataset_all_pixels saved as 'dataset_all_pixels.mat'

%% Training Support Vector Machine pipelines
load('X:\final_datasets\dataset_outdoor_relabelled.mat');                  %includes: water, vegetation, wood, rock, plastic
load('X:\final_datasets\training_dataset_relabelled.mat');                 %includes: water, vegetation, wood, plastic, sand

pipeline_svc = sdsvc(training_dataset_relabelled);


%% Classification - loading data
clear dim fns FOLD fullname info irIm training_dataset_north dataset outdoor

load('X:\Riverbank-litter\riverbank_data_ir_matched.mat');                 %Loads riverbank litter ROIs (FX17 - LAB)
load('X:\final_datasets\training_dataset_relabelled.mat');                 %Loads outdoor environment ROIs (SNAPSCAN SWIR)
load('X:\final_datasets\dataset_all_pixels.mat');                          %Loads all annotated ROIs on both outdoor environment images (SNAPSCAN SWIR)
load('E:\f5.6_25ms_01_corrected.mat');                                     %Loads image to classify
load('E:\w35ms_f5.6_60ms_north_03_corrected.mat');                         %Loads training image

%% Classification based on lab-data
%uncomment if using lab data (manual matching of bands)
%training_dataset_relabelled_matched     = training_dataset_relabelled(:,[1	3	5	7	8	9	10	11	13	14	15	16	17	18	20	22	24	25	26	27	28	30	31	32	33	35	37	40	41	42	43	45	46	47	48	49	50	52	54	55	56	57	59	60	61	62	64	65	66	67	69	71	72	73	74	75	77	78	79	80	81	83	85	87	88	89	90	92	93	95	96	97	99	100],:);
%image                                   = f56_25ms_01_corrected(:,[1	3	5	7	8	9	10	11	13	14	15	16	17	18	20	22	24	25	26	27	28	30	31	32	33	35	37	40	41	42	43	45	46	47	48	49	50	52	54	55	56	57	59	60	61	62	64	65	66	67	69	71	72	73	74	75	77	78	79	80	81	83	85	87	88	89	90	92	93	95	96	97	99	100],:);

load('X:\Data2_fold\data_ir.mat')
data2 = sdrelab(data,{  'PP_1','Plastic','PP_2','Plastic','PP_3','Plastic','PP_4','Plastic','PP_5','Plastic',...
                        'PP_6','Plastic','PP_7','Plastic','PP_8','Plastic','PP_9','Plastic','PP_10','Plastic',...
                        'PP_11','Plastic','PP_12','Plastic','PP_13','Plastic','PP_14','Plastic','PP_15','Plastic',...
                        'PP_16','Plastic','PP_17','Plastic','PP_18','Plastic','HDPE_1','Plastic','HDPE_2','Plastic','HDPE_2','Plastic',...
                        'HDPE_3','Plastic','HDPE_4','Plastic','HDPE_5','Plastic','HDPE_6','Plastic',...
                        'Organic_1','Organic','Organic_2','Organic','Organic_3','Organic',...
                        'Organic_4','Organic','Orrganic_5','Organic','Organic_5','Organic','Organic_6','Organic',...
                        'Organic_7','Organic','Organic_8','Organic','LDPE_1','Plastic',...
                        'LDPE_2','Plastic','LDPE_3','Plastic','LDPE_4','Plastic','PET_1','Plastic',...
                        'PET_2','Plastic','PET_3','Plastic','PET_4','Plastic','PET_5','Plastic','PET_6','Plastic',...
                        'PET_7','Plastic','PET_8','Plastic','PET_9','Plastic','PP_sub_1','SUB','PP_sub_2','SUB','PP_sub_3','SUB',...
                        'PP_sub_4','SUB','PP_sub_5','SUB','Other_1','OTHER','Other_2','OTHER',...
                        'Other_3','OTHER','Other_4','OTHER','Organic_sub_1','SUB','PS_sub_1','SUB',...
                        'PS_sub_2','SUB','PS_1','Plastic','PS_2','Plastic','PS_3','Plastic'});
   
% Using lab-data as training dataset (manually selected matching bands)
training_dataset_lab = [data2(:,[32	33	34	35	36	37	38	39	40	41	42	43	44	45	46	47	48	49	50	51	52	53	54	55	56	57	58	59	60	61	62	63	64	65	66	67	68	69	70	71	72	73	74	75	76	77	78	79	80	81	82	83	84	85	86	87	88	89	90	91	92	93	94	95	96	97	98	99	100	101	102	103	104	105],1);data2(:,[32	33	34	35	36	37	38	39	40	41	42	43	44	45	46	47	48	49	50	51	52	53	54	55	56	57	58	59	60	61	62	63	64	65	66	67	68	69	70	71	72	73	74	75	76	77	78	79	80	81	82	83	84	85	86	87	88	89	90	91	92	93	94	95	96	97	98	99	100	101	102	103	104	105],3); data_matched(:,:,2)];

% Selecting 10.000 random water, vegetation and plastic pixels to train
water_rand = randperm(146990,10000);
vegetation_rand = randperm(72951,10000);
plastic_rand = randperm(173036,10000);

% Create the new training dataset based on the randomly selected pixels
training_dataset_lab = [training_dataset_lab(water_rand,:,1);training_dataset_lab(vegetation_rand,:,2);training_dataset_lab(plastic_rand,:,3)];

% Create the new training dataset based on the randomly selected pixels (vegetation halved)
%training_dataset_lab = [training_dataset_lab(water_rand,:,1);training_dataset_lab(vegetation_rand,:,2).*0.5;training_dataset_lab(plastic_rand,:,3)];

% Create SVC Pipeline
pipeline_svc = sdsvc(training_dataset_relabelled);

%% Loading the algorithms based on manually matched lab/field data (SVM)
load('E:\pipeline_svc.mat')                             %only field data
load('E:\pipeline_svc_field')                           %only field data (different image)
load('E:\pipeline_svc_lab.mat')                         %lab and field combi
load('E:\pipeline_svc_onlylab_10k.mat')                 %only lab data -> 10k pixels/category instead of all (significantly reduced computation time
load('E:\pipeline_svc_onlylab_vegetation_halved.mat')   %only lab data with vegetation spectrum *.5
load('E:\colormap.mat')                                 %loading colormap for plotting

%uncomment if lab data is used to match dimensionality
%image_to_classify = f56_25ms_01_corrected(:,[1	3	5	7	8	9	10	11	13	14	15	16	17	18	20	22	24	25	26	27	28	30	31	32	33	35	37	40	41	42	43	45	46	47	48	49	50	52	54	55	56	57	59	60	61	62	64	65	66	67	69	71	72	73	74	75	77	78	79	80	81	83	85	87	88	89	90	92	93	95	96	97	99	100],:);

%uncomment if only field data is used in classification
image_to_classify = f56_25ms_01_corrected;                                 %image to classify

binsdlab = image_to_classify * pipeline_svc_field;                         %run the classification
segmenteddata=setlab(image_to_classify,binsdlab);                          %set labels of the classified image
labeledimage = sdimage(segmenteddata,'labim');                             %create sd-image from dataset
grayscale_img = im(:,:,10);                                                %create grayscale image of 1225nm band from image. Make sure variables 'im' and 'image' are the same 

%% Use this section when one of the following pipelines is used for classification:
% pipeline_svc, pipeline_svc_complemented, pipeline_svc_field
load('E:\colormap.mat'); map(3,:) = [0.6 0.22 0];

%uncomment if picture without rock is used for training:
map = [0.2 0.4 0.9; 0.2 0.7 0.1; 0.6 0.22 0; 0.8 0.1 0.8; 0.9 0.8 0.4]

classified_image = labeloverlay(grayscale_img,labeledimage,'IncludedLabels',[1,2,3,4,5],'colormap',map); 
figure('DefaultAxesFontSize',16); set(gcf,'Position',[150 200 1700 700])
subplot(1,3,1); imshow(imread('E:\image_rgb_dim.png')); title('RGB')
subplot(1,3,2); imshow(imrotate(im(:,:,20),-90)); title('IR - Grayscale (1225 nm)')
subplot(1,3,3); imshow(imrotate(classified_image,-90)); title('Classified image');

hold on; 
x0 = get(gca,'xlim');
y0 = get(gca,'ylim');
plot(0,0,'MarkerEdgeColor',[0.2 0.4 0.9],'MarkerFaceColor',[0.2 0.4 0.9],'marker','s','MarkerSize',20,'LineStyle','none');
plot(0,0,'MarkerEdgeColor',[0.2 0.7 0.1],'MarkerFaceColor',[0.2 0.7 0.1],'marker','s','MarkerSize',20,'LineStyle','none');
plot(0,0,'MarkerEdgeColor',[0.6 0.4 0.0],'MarkerFaceColor',[0.6 0.4 0.0],'marker','s','MarkerSize',20,'LineStyle','none');
plot(0,0,'MarkerEdgeColor',[0.6 0.6 0.6],'MarkerFaceColor',[0.6 0.6 0.6],'marker','s','MarkerSize',20,'LineStyle','none');
plot(0,0,'MarkerEdgeColor',[0.8 0.1 0.8],'MarkerFaceColor',[0.8 0.1 0.8],'marker','s','MarkerSize',20,'LineStyle','none');
plot(0,0,'MarkerEdgeColor',[0.9 0.8 0.4],'MarkerFaceColor',[0.9 0.8 0.4],'marker','s','MarkerSize',20,'LineStyle','none');
legend('Water','Vegetation','Wood','Rock','Plastic','Sand','Location','South','NumColumns',2)

%% Use this section when one of the following pipelines is used for classification:
% pipeline_svc_lab, pipeline_svc_onlylab, pipeline_svc_onlylab_10k,
% pipeline_svc_onlylab_vegetation_halved, %pipeline_svc_onlylab_vegetation
map2 = map([1 2 5],:);

classified_image = labeloverlay(grayscale_img,labeledimage,'IncludedLabels',[1,2,3],'colormap',map2);
figure('DefaultAxesFontSize',16); set(gcf,'Position',[150 200 1700 700])
subplot(1,3,1); imshow(imread('E:\image_rgb_dim.png')); title('RGB')
subplot(1,3,2); imshow(imrotate(im(:,:,20),-90)); title('IR - Grayscale (1225 nm)')
subplot(1,3,3); imshow(imrotate(classified_image,-90)); title('Classified image')

hold on; 
x0 = get(gca,'xlim');
y0 = get(gca,'ylim');
plot(0,0,'MarkerEdgeColor',[0.2 0.4 0.9],'MarkerFaceColor',[0.2 0.4 0.9],'marker','s','MarkerSize',20,'LineStyle','none');
plot(0,0,'MarkerEdgeColor',[0.2 0.7 0.1],'MarkerFaceColor',[0.2 0.7 0.1],'marker','s','MarkerSize',20,'LineStyle','none');
plot(0,0,'MarkerEdgeColor',[0.8 0.1 0.8],'MarkerFaceColor',[0.8 0.1 0.8],'marker','s','MarkerSize',20,'LineStyle','none');

legend('Water','Vegetation','Plastic','Location','South','NumColumns',2)

%% Plotting different spectral signatures of LAB (riverbank v.s. pristine) and field (riverbank v.s. pristine) spectra
close all
load('E:\training_dataset_lab.mat');
load('E:\mean_data_lab_intp.mat');
load('E:\plastic_pristine.mat');
load('X:\final_datasets\dataset_outdoor_relabelled.mat');                  %includes: water, vegetation, wood, rock, plastic
load('X:\final_datasets\training_dataset_relabelled.mat');                 %includes: water, vegetation, wood, plastic, sand
load('E:\mean_data.mat');

lab_water               = mean(getdata(training_dataset_lab(:,:,1)));
lab_vegetation          = mean(getdata(training_dataset_lab(:,:,2)));
lab_plastics_pristine	= object_mean_ir;
lab_plastics_riverbank  = mean_data_lab_intp(2,:);

%interpolating to match spectral range
t = linspace(1156,1674,74); ti = linspace(1156,1674,100);
lab_water = interp1(t,lab_water,ti);

lab_vegetation = interp1(t,lab_vegetation,ti);
lab_plastics_pristine = interp1(t,lab_plastics_pristine,ti);

field_water = mean(getdata(dataset_all_pixels(:,:,1)));
field_vegetation = mean(getdata(dataset_all_pixels(:,:,2)));
field_plastics = mean(getdata(dataset_all_pixels(:,:,5)));

figure('DefaultAxesFontSize',12); set(gcf,'Position',[200 200 1275 500]); 
sgtitle('Lab and field based reflectance signatures of water, vegetation and plastic','FontSize',18);

subplot(1,3,1)
plot(lab_water,':','color',[0.2 0.4 0.9],'LineWidth',2); hold on;
plot(field_water,'color',[0.2 0.4 0.9],'LineWidth',1.5);
xlim([1 100]); xticks([10 28 48 66 86]); ylim([0 1.0]); xticklabels([1200 1300  1400  1500  1600]);
legend('Water (lab)','Water (field)','location','north')
ylabel('Ref. reflectance (-)'); xlabel('Wavelength (nm)')

subplot(1,3,2)
plot(lab_vegetation,':','color',[0.2 0.7 0.1],'LineWidth',2); hold on;
plot(field_vegetation,'color',[0.2 0.7 0.1],'LineWidth',1.5);
xlim([1 100]); xticks([10 28 48 66 86]); ylim([0 1.0]); xticklabels([1200 1300  1400  1500  1600]);
legend('Vegetation (lab)','Vegetation (field)','location','northeast')
xlabel('Wavelength (nm)')

subplot(1,3,3)
plot(lab_plastics_pristine,'-.','color',[0.8 0.1 0.8],'LineWidth',1.5); hold on;
plot(lab_plastics_riverbank,':','color',[0.8 0.1 0.8],'LineWidth',2);
plot(field_plastics,'color',[0.8 0.1 0.8],'LineWidth',1.5);
xlim([1 100]); xticks([10 28 48 66 86]); ylim([0 1.0]); xticklabels([1200 1300  1400  1500  1600]);
legend('Pristine plastics (lab)','Riverbank plastics (lab)','Riverbank plastics (field)','location','north')
xlabel('Wavelength (nm)')

%% Classification and visualisation (SVM)
load('E:\w35ms_f5.6_60ms_north_03_corrected.mat')
load('E:\W30ms_f5.6_40ms_corrected.mat')
load('E:\f5.6_25ms_01_corrected.mat')
load('E:\W35ms_f5.6_35ms_north_corrected.mat')
load('E:\colormap.mat')
load('E:\pipeline_svc.mat')
load('E:\pipeline_svc_complemented.mat')

image_to_classify = f56_25ms_01_corrected;
binsdlab = image_to_classify * pipeline_svc_complemented;
segmenteddata=setlab(image_to_classify,binsdlab);
labeledimage = sdimage(segmenteddata,'labim');
grayscale_img = im(:,:,30);
classified_image = labeloverlay(grayscale_img,labeledimage,'IncludedLabels',[1,2,3,4,5,6],'colormap',map);

figure('DefaultAxesFontSize',16); set(gcf,'Position',[150 200 1700 700])
subplot(1,3,1); imshow(imread('E:\image_rgb_dim.png')); title('RGB')
subplot(1,3,2); imshow(imrotate(im(:,:,20),-90)); title('IR - Grayscale (1225 nm)')
subplot(1,3,3); imshow(imrotate(classified_image,-90)); title('Classified image')

hold on; 
x0 = get(gca,'xlim');
y0 = get(gca,'ylim');
plot(0,0,'MarkerEdgeColor',[0.2 0.4 0.9],'MarkerFaceColor',[0.2 0.4 0.9],'marker','s','MarkerSize',20,'LineStyle','none');
plot(0,0,'MarkerEdgeColor',[0.2 0.7 0.1],'MarkerFaceColor',[0.2 0.7 0.1],'marker','s','MarkerSize',20,'LineStyle','none');
plot(0,0,'MarkerEdgeColor',[0.6 0.4 0.0],'MarkerFaceColor',[0.6 0.4 0.0],'marker','s','MarkerSize',20,'LineStyle','none');
plot(0,0,'MarkerEdgeColor',[0.6 0.6 0.6],'MarkerFaceColor',[0.6 0.6 0.6],'marker','s','MarkerSize',20,'LineStyle','none');
plot(0,0,'MarkerEdgeColor',[0.8 0.1 0.8],'MarkerFaceColor',[0.8 0.1 0.8],'marker','s','MarkerSize',20,'LineStyle','none');
plot(0,0,'MarkerEdgeColor',[0.9 0.8 0.4],'MarkerFaceColor',[0.9 0.8 0.4],'marker','s','MarkerSize',20,'LineStyle','none');
legend('Water','Vegetation','Wood','Rock','Plastic','Sand','Location','South','NumColumns',2)

%% Using only outside data
close all
numEndmembers = 5;

for i = 1:5
    mean_data(i,:) = mean(getdata(training_dataset_relabelled(:,:,i)));
end

sam_score = zeros(size(im,1),size(im,2),numEndmembers);
for i = 1:numEndmembers
    sam_score(:,:,i) = sam(im,mean_data(i,:));
end

sid_score = zeros(size(im,1),size(im,2),numEndmembers);
for i = 1:numEndmembers
    sid_score(:,:,i) = sid(im,mean_data(i,:));
end

sidsam_score = zeros(size(im,1),size(im,2),numEndmembers);
for i = 1:numEndmembers
    sidsam_score(:,:,i) = sidsam(im,mean_data(i,:));
end

colormap_no_rock =   [ 0.2 0.4 0.9                                         % Create colormap for the figure
                       0.2 0.7 0.1
                       0.6 0.4 0.0
                       0.9 0.8 0.4
                       0.8 0.1 0.8
                    ];
                

treshold = 0.1308; %set threshold for SAM: 15 deg = 0.261 rad, 10 deg = 0.1745 rad, 7.5 deg = 0.1308 rad. put '9' if no treshold is used
sam_score(sam_score>treshold) =  9999;

[~,sam_matchingIndx] = min(sam_score,[],3);
[~,sid_matchingIndx] = min(sid_score,[],3);
[~,sidsam_matchingIndx] = min(sidsam_score,[],3);

for i = 1:size(sam_score,3)
   sam_matchingIndx(sam_score(:,:,i) == 9999 & sam_matchingIndx == i) = 0;
end
%% Plotting classification using outdoor data
figure('DefaultAxesFontSize',13); load('E:\colormap.mat'); set(gcf,'Position',[0 0 512 640])
%imagesc(rot90(rot90(rot90((sam_matchingIndx))))); colormap([0 0 0; colormap_no_rock]); set(gca,'xtick',[]); set(gca,'ytick',[]); %use for sam
imagesc(rot90(rot90(rot90((sam_matchingIndx))))); colormap([colormap_no_rock]); set(gca,'xtick',[]); set(gca,'ytick',[]); %use for sid and sidsam
hold on; 
x0 = get(gca,'xlim');
y0 = get(gca,'ylim');
plot(0,0,'MarkerEdgeColor',[0.2 0.4 0.9],'MarkerFaceColor',[0.2 0.4 0.9],'marker','s','MarkerSize',20,'LineStyle','none');
plot(0,0,'MarkerEdgeColor',[0.2 0.7 0.1],'MarkerFaceColor',[0.2 0.7 0.1],'marker','s','MarkerSize',20,'LineStyle','none');
plot(0,0,'MarkerEdgeColor',[0.6 0.4 0.0],'MarkerFaceColor',[0.6 0.4 0.0],'marker','s','MarkerSize',20,'LineStyle','none');
plot(0,0,'MarkerEdgeColor',[0.6 0.6 0.6],'MarkerFaceColor',[0.6 0.6 0.6],'marker','s','MarkerSize',20,'LineStyle','none');
plot(0,0,'MarkerEdgeColor',[0.8 0.1 0.8],'MarkerFaceColor',[0.8 0.1 0.8],'marker','s','MarkerSize',20,'LineStyle','none');
plot(0,0,'MarkerEdgeColor',[0.9 0.8 0.4],'MarkerFaceColor',[0.9 0.8 0.4],'marker','s','MarkerSize',20,'LineStyle','none');
legend('Water','Vegetation','Wood','Rock','Plastic','Sand','Location','South','NumColumns',2)

figure; 
subplot(1,2,1); imagesc(rot90(rot90(rot90((sid_matchingIndx))))); colormap(colormap_no_rock)
subplot(1,2,2); imagesc(rot90(rot90(rot90((sidsam_matchingIndx))))); colormap(colormap_no_rock)

% sam_montage = cat(3,sam_matchingIndx,sid_matchingIndx,jm_matchingIndx,ns3_matchingIndx);
% figure; montage(sam_montage,'Size',[2 2],'BorderSize',[10 10]); colormap(map);colorbar(); title('SAM classifiers')

%% Using only indoor data
clear sam_score sid_score sidsam_score sam_matchingIndx sid_matchingIndx sidsam_matchingIndx %clear all scores and indices to ensure correct calculation
load('X:\mean_data_lab_intp.mat')
load('E:\mean_data.mat');

numEndmembers = 3;                                                         % Specify number of endmembers = 3 = water, vegetation, plastic
map_outside = [0.2 0.4 0.9; 0.2 0.7 0.1; 0.8 0.1 0.8];                     % Create colormap for water, vegetation, plastic

sam_score = zeros(size(im,1),size(im,2),numEndmembers);
sam_score(:,:,1) = sam(im,mean_data_lab_intp(9,:));                        % Calculate sam score
sam_score(:,:,2) = sam(im,mean_data_lab_intp(8,:));
sam_score(:,:,3) = sam(im,mean_data_lab_intp(2,:));

sid_score = zeros(size(im,1),size(im,2),numEndmembers);
sid_score(:,:,1) = sid(im,mean_data_lab_intp(9,:));                        % Calculate sid score
sid_score(:,:,2) = sid(im,mean_data_lab_intp(8,:));
sid_score(:,:,3) = sid(im,mean_data_lab_intp(2,:));

sidsam_score = zeros(size(im,1),size(im,2),numEndmembers);
sidsam_score(:,:,1) = sidsam(im,mean_data_lab_intp(9,:));                  % Calculate sid score
sidsam_score(:,:,2) = sidsam(im,mean_data_lab_intp(8,:));
sidsam_score(:,:,3) = sidsam(im,mean_data_lab_intp(2,:));

treshold = 0.1308; %set threshold for SAM: 15 deg = 0.261 rad, 10 deg = 0.1745 rad, 7.5 deg = 0.1308 rad. put '9' if no treshold is used
sam_score(sam_score>treshold) =  9999;

[~,sam_matchingIndx] = min(sam_score,[],3);
[~,sid_matchingIndx] = min(sid_score,[],3);
[~,sidsam_matchingIndx] = min(sidsam_score,[],3);

for i = 1:size(sam_score,3)
   sam_matchingIndx(sam_score(:,:,i) == 9999 & sam_matchingIndx == i) = 0;
end


%% Plotting classification using lab data
figure('DefaultAxesFontSize',13); numEndmembers = 3; map_outside = [0.2 0.4 0.9; 0.2 0.7 0.1; 0.8 0.1 0.8]; set(gcf,'Position',[0 0 512 640])
%imagesc(rot90(rot90(rot90((sam_matchingIndx))))); colormap([0 0 0;
%map_outside]); set(gca,'xtick',[]); set(gca,'ytick',[]); %use for sam when having a cone of uncertainty 
imagesc(rot90(rot90(rot90((sam_matchingIndx))))); colormap([map_outside]); set(gca,'xtick',[]); set(gca,'ytick',[]); %use for sid and sidsam
hold on; x0 = get(gca,'xlim'); y0 = get(gca,'ylim');
plot(0,0,'MarkerEdgeColor',[0.2 0.4 0.9],'MarkerFaceColor',[0.2 0.4 0.9],'marker','s','MarkerSize',20,'LineStyle','none');
plot(0,0,'MarkerEdgeColor',[0.2 0.7 0.1],'MarkerFaceColor',[0.2 0.7 0.1],'marker','s','MarkerSize',20,'LineStyle','none');
plot(0,0,'MarkerEdgeColor',[0.8 0.1 0.8],'MarkerFaceColor',[0.8 0.1 0.8],'marker','s','MarkerSize',20,'LineStyle','none');
legend('Water','Vegetation','Plastic','Location','South','NumColumns',1)

figure; 
subplot(1,3,1); imagesc(rot90(rot90(rot90((sam_matchingIndx))))); colormap(map_outside)
subplot(1,3,2); imagesc(rot90(rot90(rot90((sid_matchingIndx))))); colormap(map_outside)
subplot(1,3,3); imagesc(rot90(rot90(rot90((sidsam_matchingIndx))))); colormap(map_outside)

%% Accuracy assessment
load('X:\Riverbank-litter\training_data_outdoor.mat')                      % Loading training data
load('E:\training_dataset_north.mat')                                      % Loading training data (Figure 3d)
        
training_data           =   sdrelab(training_data,{'water','water','vegetation','vegetation','wood','wood','rock','rock','sample_1','plastic','sample_2','plastic','sample_3','plastic','sample_4','plastic','sample_5','plastic','sample_6','plastic','sample_7','plastic','sample_8','plastic','sample_9','plastic','sample_10','plastic','sample_11','plastic','sample_12','plastic','sample_13','plastic','sample_13','plastic','sample_14','plastic','sample_15','plastic','sample_16','plastic','sample_17','plastic','sample_18','plastic','sample_19','plastic','sample_20','plastic','sample_21','plastic','sample_22','plastic','sample_23','plastic','sample_24','plastic','sample_25','plastic','sample_26','plastic'});
training_data_north     =   sdrelab(training_data_north,{'water','water','vegetation','vegetation','wood','wood','rock','rock','sample_1','plastic','sample_2','plastic','sample_3','plastic','sample_4','plastic','sample_5','plastic','sample_6','plastic','sample_7','plastic','sample_8','plastic','sample_9','plastic','sample_10','plastic','sample_11','plastic','sample_12','plastic','sample_13','plastic','sample_13','plastic','sample_14','plastic','sample_15','plastic','sample_16','plastic','sample_17','plastic','sample_18','plastic','sample_19','plastic','sample_20','plastic','sample_21','plastic','sample_22','plastic','sample_23','plastic','sample_24','plastic','sample_25','plastic'});

%% Training_data - accuracy assessment
close all
water_region = sdimage(training_data(:,1,2),'matrix');water_region = water_region>0;
vegetation_region = sdimage(training_data(:,1,3),'matrix');vegetation_region = vegetation_region>0;
wood_region = sdimage(training_data(:,1,4),'matrix');wood_region = wood_region>0;
rock_region = sdimage(training_data(:,1,5),'matrix');rock_region = rock_region>0;
plastic_region = sdimage(training_data(:,1,6),'matrix');plastic_region = plastic_region>0;
sand_region = logical(zeros(size(water_region)));

mask_training_data = zeros(size(water_region));
mask_training_data(water_region==1)=1;
mask_training_data(vegetation_region==1)=2;
mask_training_data(wood_region==1)=3;
mask_training_data(rock_region==1)=4;
mask_training_data(plastic_region==1)=5;
mask_training_data(sand_region==1)=6;

classLabels = {'Water','Vegetation','Wood','Rock','Plastic'};
confusion_matrix = confusionmat(reshape(mask_training_data,1,[]),reshape(sam_matchingIndx,1,[]));
confusion_matrix = confusion_matrix(2:6,2:6);
cm = confusionchart(confusion_matrix,classLabels); cm.RowSummary = 'row-normalized'; cm.ColumnSummary = 'column-normalized';

% Calculate percentage of pixels not classified in SAM
not_classified = sam_matchingIndx == 0;
missed_pixels = sum(not_classified == 1 & mask_training_data == 5);
total_pixels = sum(mask_training_data == 5);
percentage_missed = sum(missed_pixels)/sum(total_pixels)*100;

missed_text = ['SAM Threshold degree is: ',num2str(180/pi*treshold) ' degrees. Total plastic pixels is: ',num2str(sum(total_pixels)),'. Missed pixels with this threshold is: ',num2str(sum(missed_pixels)),'. Thats a percentage of: ',num2str(percentage_missed),'%'];
missed_text

%% Training_data
close all
water_region = sdimage(training_data(:,1,2),'matrix');water_region = water_region>0;
vegetation_region = sdimage(training_data(:,1,3),'matrix');vegetation_region = vegetation_region>0;
wood_region = sdimage(training_data(:,1,4),'matrix');wood_region = wood_region>0;
rock_region = sdimage(training_data(:,1,5),'matrix');rock_region = rock_region>0;
plastic_region = sdimage(training_data(:,1,6),'matrix');plastic_region = plastic_region>0;
sand_region = logical(zeros(size(water_region)));

mask_training_data = zeros(size(water_region));
mask_training_data(water_region==1)=1;
mask_training_data(vegetation_region==1)=2;
mask_training_data(wood_region==1)=3;
mask_training_data(rock_region==1)=4;
mask_training_data(plastic_region==1)=5;
mask_training_data(sand_region==1)=6;

classLabels = {'Water','Vegetation','Wood','Rock','Plastic','Sand'};
confusion_matrix = confusionmat(reshape(mask_training_data,1,[]),reshape(sam_matchingIndx,1,[]));
confusion_matrix = confusion_matrix(2:7,2:7);
cm = confusionchart(confusion_matrix,classLabels); cm.RowSummary = 'row-normalized'; cm.ColumnSummary = 'column-normalized';

%% Training_data_north
water_region = sdimage(training_data_north(:,1,2),'matrix');water_region = water_region>0;
vegetation_region = sdimage(training_data_north(:,1,3),'matrix');vegetation_region = vegetation_region>0;
wood_region = sdimage(training_data_north(:,1,4),'matrix');wood_region = wood_region>0;
sand_region = sdimage(training_data_north(:,1,5),'matrix');sand_region = sand_region>0;
plastic_region = sdimage(training_data_north(:,1,6),'matrix');plastic_region = plastic_region>0;
rock_region = logical(zeros(size(rock_region)));

mask_training_data_north = zeros(size(water_region));
mask_training_data_north(water_region==1)=1;
mask_training_data_north(vegetation_region==1)=2;
mask_training_data_north(wood_region==1)=3;
mask_training_data_north(rock_region==1)=4;
mask_training_data_north(plastic_region==1)=5;
mask_training_data_north(sand_region==1)=6;

classLabels = {'Water','Vegetation','Wood','Rock','Plastic','Sand'};
confusion_matrix = confusionmat(reshape(mask_training_data,1,[]),reshape(sid_matchingIndx,1,[]));
confusion_matrix = confusion_matrix(2:7,2:7);
cm = confusionchart(confusion_matrix,classLabels); cm.RowSummary = 'row-normalized'; cm.ColumnSummary = 'column-normalized';
figure; imagesc(mask_training_data_north)
%% Training_data only lab

water_region = sdimage(training_data(:,1,2),'matrix');water_region = water_region>0;
vegetation_region = sdimage(training_data(:,1,3),'matrix');vegetation_region = vegetation_region>0;
plastic_region = sdimage(training_data(:,1,6),'matrix');plastic_region = plastic_region>0;

mask_training_data_labfield = zeros(size(water_region));
mask_training_data_labfield(water_region==1)=1;
mask_training_data_labfield(vegetation_region==1)=2;
mask_training_data_labfield(plastic_region==1)=3;

classLabels = {'Water','Vegetation','Plastic'};
%confusion_matrix = confusionmat(reshape(mask_training_data_labfield,1,[]),reshape(double(labeledimage),1,[]));
confusion_matrix = confusionmat(reshape(mask_training_data_labfield,1,[]),reshape(sam_matchingIndx,1,[]));

confusion_matrix = confusion_matrix(2:4,2:4);
figure; cm = confusionchart(confusion_matrix,classLabels); cm.RowSummary = 'row-normalized'; cm.ColumnSummary = 'column-normalized';

% Calculate percentage of pixels not classified in SAM
not_classified = sam_matchingIndx == 0;
missed_pixels = sum(not_classified == 1 & mask_training_data_labfield == 3);
total_pixels = sum(mask_training_data_labfield == 3);
percentage_missed = sum(missed_pixels)/sum(total_pixels)*100;

missed_text = ['SAM Threshold degree is: ',num2str(180/pi*treshold) ' degrees. Total plastic pixels is: ',num2str(sum(total_pixels)),'. Missed pixels with this threshold is: ',num2str(sum(missed_pixels)),'. Thats a percentage of: ',num2str(percentage_missed),'%'];
missed_text

%% Plotting classification on top of masks
close all;
map_no_sand =   [  1 1 1
           0.2 0.4 0.9 
           0.2 0.7 0.1
           0.6 0.4 0.0
           0.6 0.6 0.6
           0.8 0.1 0.8
        ];
    
map_no_rock =   [  0    0   0
           0.2 0.4 0.9 
           0.2 0.7 0.1
           0.6 0.4 0.0
           0.8 0.1 0.8
           0.8 0.1 0.8                
           0.9 0.8 0.4
        ];

alpha_vals = mask_training_data > 0;
alpha_vals_north = mask_training_data_north > 0;
   
figure1 = figure('DefaultAxesFontSize',16);
f = gcf; f.Position(3) = f.Position(3) * 1.25; f.Position(4) = f.Position(4) * 2; f.Position(1) = 0; f.Position(2) = 0;
ax1 = axes('Parent',figure1); ax2 = axes('Parent',figure1);
set(ax1,'Visible','off'); set(ax2,'Visible','off');
I = imagesc(rot90(rot90(rot90(mask_training_data_north))),'Parent',ax2); colormap(map_no_rock); alpha = rot90(rot90(rot90(alpha_vals_north-.2))); ; axis off
imshow(rot90(rot90(rot90(im(:,:,10)))),'Parent',ax1);
set(I,'AlphaData',alpha);

%% SAM using field data and lab data (accuracy/thresold)
f_tresholds = [7.5 10 15];
f_missed_pixels_arr = [42.17 15.61 3.70];

l_tresholds = [7.5 10 15];
l_missed_pixels_arr = [85.69 41.38 8.016];

figure('DefaultAxesFontSize',14); 
plot(f_tresholds,f_missed_pixels_arr,'k','LineWidth',1.5); ylabel('% missed pixels'); xlabel('SAM Cone of Uncertainty (degrees)'); hold on
plot(l_tresholds,l_missed_pixels_arr,'r','LineWidth',1.5); 
legend('Field Data','Lab Data');title('n = 8.370 plastic pixels')