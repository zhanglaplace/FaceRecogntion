% --------------------------------------------------------
% Copyright (c) Weiyang Liu, Yandong Wen
% Licensed under The MIT License [see LICENSE for details]
%
% Intro:
% This script is used to evaluate the performance of the trained model on LFW dataset.
% We perform 10-fold cross validation, using cosine similarity as metric.
% More details about the testing protocol can be found at http://vis-www.cs.umass.edu/lfw/#views.
% 
% Usage:
% cd $SPHEREFACE_ROOT/test
% run code/evaluation.m
% --------------------------------------------------------

function evaluation_ytf()

clear;clc;close all;

%% load txt
pair_list  = 'splits_no_header.txt';
pair_fid = fopen(pair_list,'r');
C = textscan(pair_fid,'%d %d %s %s %d %d');
fclose(pair_fid);



%% caffe setttings
matCaffe = '/home/zf/deeplearning/caffe/matlab';
addpath(matCaffe);
gpu = true;
if gpu
   gpu_id = 0;
   caffe.set_mode_gpu();
   caffe.set_device(gpu_id);
else
   caffe.set_mode_cpu();
end
caffe.reset_all();

model   = '/data/youtubeface/face_deploy_amsoftmax.prototxt';
weights = '/home/zf/deeplearning/caffe/face_example/face-recognition/AMSoftmax/ms_1m/face_snapshot/ms1m_train_test_iter_150000.caffemodel';
net     = caffe.Net(model, weights, 'test');
%net.save('result/sphereface_model.caffemodel');


load ytf_eval_list.mat
load ytf_subset_map.mat


%% extract feature
feature_dim = 512;
batch_size = 100;
mean_value = 128;
scale = 0.0078125;
ROIx = 1:96;
ROIy = 1:112;
height = length(ROIx);
width = length(ROIy);
total_image = length(image_list);
total_iter = ceil(total_image / batch_size);
features = zeros(feature_dim,total_iter*batch_size);
image_p = 1;
for i=1:total_iter
    fprintf('%d/%d\n',i, total_iter);
    J = zeros(height,width,3,batch_size,'single');
    for j = 1 : batch_size
        if image_p <= total_image
            I = imread(image_list{image_p});
            I = permute(I,[2 1 3]);
            I = I(:,:,[3 2 1]);
            I = I(ROIx,ROIy,:);
            I = single(I) - mean_value;
            J(:,:,:,j) = I*scale;
            image_p = image_p + 1;
        end;
    end;
    f1 = net.forward({J});
    f1 = f1{1};
    features(:,(i-1)*batch_size+1:i*batch_size) = f1;
end;
save('feature_amsoftmax_ms1m.mat','features','-v7.3');
%load ../result/ytf/feature_center.mat
%% accuracy and roc curve


%for i=1:length(C{1})
 %   C{3}{i} = C{3}{i}(1:end-1);
  %  C{4}{i} = C{4}{i}(1:end-1);
%end;

accuracies = zeros(10,1);
distance_dist = cell(10,1);
decisions_mean = zeros(5000,1);
for cross = 1:10
    fprintf('%d-th cross validation, collect mean...', cross); 
    train_set1 = C{3}(C{1}~=cross);
    train_set2 = C{4}(C{1}~=cross);
    feature_mean = zeros(size(features,1),1);
    feature_count = 0;
    feature1_cross = cell(length(train_set1),1);
    feature2_cross = cell(length(train_set1),1);
    for i=1:length(train_set1)
        feature_index = subset_map(train_set1{i});
        feature1 = features(:,feature_index(1):feature_index(2));
        feature_mean = feature_mean + mean(feature1,2);
        feature_index = subset_map(train_set2{i});
        feature2 = features(:,feature_index(1):feature_index(2));
        feature_mean = feature_mean + mean(feature2,2);
        feature_count = feature_count + 2;
        feature1_cross{i} = feature1;
        feature2_cross{i} = feature2;
        if mod(i,450) == 0
            fprintf('%d.', int32(i / 450)); 
        end;
    end;
    fprintf('done.\n');
    fprintf('compute distance...');
    feature_mean = feature_mean / feature_count;
    distance_cross = zeros(length(C{3}),length(-1:0.02:1) - 1);
    mean_distances = zeros(length(C{3}),1);
    for i=1:length(C{3})
        feature_index = subset_map(C{3}{i});
        feature1 = features(:,feature_index(1):feature_index(2));
        feature1 = bsxfun(@minus, feature1, feature_mean);
        feature1 = bsxfun(@rdivide, feature1, sqrt(sum(feature1.^2)));
        feature_index = subset_map(C{4}{i});
        feature2 = features(:,feature_index(1):feature_index(2));
        feature2 = bsxfun(@minus, feature2, feature_mean);
        feature2 = bsxfun(@rdivide, feature2, sqrt(sum(feature2.^2)));
       
        distance_matrix = feature1' * feature2;
        histo = histcounts(distance_matrix(:),-1:0.02:1,'Normalization','probability');
        distance_cross(i,:) = histo;
        mean_distances(i) = mean(distance_matrix(:));
        if mod(i,500) == 0
            fprintf('%d.', int32(i / 500)); 
        end;
    end;
    distance_dist{cross} = distance_cross;
    fprintf('done.\n');
    cmd = [' -t 0 -h 0'];
    model = libsvmtrain(double(C{5}(C{1}~=cross)),mean_distances(C{1}~=cross),cmd);
    [class, accuracy, deci] = libsvmpredict(double(C{5}(C{1}==cross)),mean_distances(C{1}==cross),model);
%     cmd = [' -t 2 -h 0'];
%     model = svmtrain(double(C{5}(C{1}~=cross)),distance_cross(C{1}~=cross,:),cmd);
%     [class, accuracy, deci] = svmpredict(double(C{5}(C{1}==cross)),distance_cross(C{1}==cross,:),model);
    fprintf('%d th-fold accuracy:%.4f\n', cross, accuracy(1));
    accuracies(cross) = accuracy(1);
    decisions_mean((cross-1)*500+1:cross*500) = deci;
end;
fprintf('accuracy by 10-fold evalution:%.4f\n', mean(accuracies));
roc_curve(decisions_mean, C{5}*2-1);
end



function [image_list] = get_image_list_in_folder(folder)
    root_list = dir(folder);
    root_list = root_list(3:end);
    image_list = {};
    for i=1:length(root_list)
        if root_list(i).isdir
            sub_list = get_image_list_in_folder(fullfile(folder,root_list(i).name));
            image_list = [image_list;sub_list];
        else
            [~, ~, c] = fileparts(root_list(i).name);
            if strcmp(c,'.png') == 0 && strcmp(c,'.jpg') == 0 && strcmp(c,'.bmp') == 0 && strcmp(c,'.jpeg') == 0 ...
                && strcmp(c,'.PNG') == 0 && strcmp(c,'.JPG') == 0 && strcmp(c,'.BMP') == 0 && strcmp(c,'.JPEG') == 0
                continue;
            end;
            image_list = [image_list;fullfile(folder,root_list(i).name)];
        end;
    end;
end

function auc = roc_curve(deci,label_y) %?ci=wx+b, label_y, true label
    [val,ind] = sort(deci,'descend');
    roc_y = label_y(ind);
    stack_x = cumsum(roc_y == -1)/sum(roc_y == -1);
    stack_y = cumsum(roc_y == 1)/sum(roc_y == 1);
    auc = sum((stack_x(2:length(roc_y),1)-stack_x(1:length(roc_y)-1,1)).*stack_y(2:length(roc_y),1))
 
        %Comment the above lines if using perfcurve of statistics toolbox
        %[stack_x,stack_y,thre,auc]=perfcurve(label_y,deci,1);
    loglog(stack_x,stack_y);
%     plot(stack_x,stack_y);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(['ROC curve of (AUC = ' num2str(auc) ' )']);
end


function bestThreshold = getThreshold(scores, flags, thrNum)
    accuracys  = zeros(2*thrNum+1, 1);
    thresholds = (-thrNum:thrNum) / thrNum;
    for i = 1:2*thrNum+1
        accuracys(i) = getAccuracy(scores, flags, thresholds(i));
    end
    bestThreshold = mean(thresholds(accuracys==max(accuracys)));
end

function accuracy = getAccuracy(scores, flags, threshold)
    accuracy = (length(find(scores(flags==1)>threshold)) + ...
                length(find(scores(flags~=1)<threshold))) / length(scores);
end


function[subset_map] = create_folder_map(image_list)
    
    subset_map = containers.Map;
    [a,b,c] = fileparts(image_list{1});
    last_folder_all = a;
    folder_names = strsplit(a,'/');
    last_folder = [folder_names{end-1} '/' folder_names{end}];
    subset_map(last_folder) = [1 0];
    for i=2:length(image_list)
        [a,b,c] = fileparts(image_list{i});
        folder_names = strsplit(a,'/');
        if strcmp(last_folder_all, a) == 0
            start_end = subset_map(last_folder);
            subset_map(last_folder) = [start_end(1) i-1];
            last_folder = [folder_names{end-1} '/' folder_names{end}];
            subset_map(last_folder) = [i 0];
        end;
        last_folder_all = a;
    end;
    start_end = subset_map(last_folder);
    subset_map(last_folder) = [start_end(1) length(image_list)];

end