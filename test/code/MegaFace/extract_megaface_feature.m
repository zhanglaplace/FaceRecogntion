function extract_megaface_feature()
   
    matcaffe = '/home/zf/deeplearning/caffe/matlab';
    addpath(matcaffe);
    caffe.reset_all();
    caffe.set_mode_gpu();
    gpu_id = 0;  % we will use the first gpu in this demo
    caffe.set_device(gpu_id);

    batch_size = 50;
    feature_dim = 512;
    mean_value = 128;
    scale = 0.0078125;
    ROIx = 1:96;
    ROIy = 1:112;
    height = length(ROIx);
    width = length(ROIy);

    feature_suffix = '_hard64nofd';
    aligned_image_folder = '/home/zf/deeplearning/datasets/face-recognition/megaface/MegaFace-112x96/';
    feature_folder = ['/home/zf/deeplearning/datasets/face-recognition/megaface/MegaFace_feature' feature_suffix '/'];
    if exist(feature_folder, 'dir')==0
        mkdir(feature_folder);
    end;

    load ../../../preprocess/code/MegaFace/megaface_test_crop_list_100w.mat
    image_list = megaface_test_crop_list_100w;
    


    image_p = 1;
    total_image = length(image_list);
    total_iter = ceil((total_image - image_p + 1) / batch_size);

    model   = '/data/youtubeface/face_deploy_amsoftmax_batch50.prototxt';
    weights = '/home/zf/deeplearning/caffe/face_example/face-recognition/AMSoftmax/vggface/face_snapshot/vgg_train_test_iter_70000.caffemodel';
    net     = caffe.Net(model, weights, 'test');

    % features = zeros(feature_dim, total_iter * batch_size);
    for i=1:total_iter
        if mod(i,100) == 1
            fprintf('%d/%d\n',i, total_iter);
        end;
        J = zeros(height,width,3,batch_size,'single');
        feature_names = cell(batch_size,1);
        for j = 1 : batch_size
            if image_p <= total_image
                I = imread(image_list{image_p});
                if size(I, 3) < 3
                   I(:,:,2) = I(:,:,1);
                   I(:,:,3) = I(:,:,1);
                end
                I = permute(I,[2 1 3]);
                I = I(:,:,[3 2 1]);
                I = I(ROIx,ROIy,:);
                I = single(I) - mean_value;
                J(:,:,:,j) = I*scale;
                feature_names{j} = [strrep(image_list{image_p},aligned_image_folder, feature_folder) feature_suffix '.bin'];
                image_p = image_p + 1;
            end;
        end;
        f1 = net.forward({J});
        feature = squeeze(f1{1});
    %     features(:,(i-1)*batch_size+1:i*batch_size) = feature;
        for j = 1 : batch_size
            if ~isempty(feature_names{j})
                [file_folder, file_name, file_ext] = fileparts(feature_names{j});
                if exist(file_folder,'dir')==0
                    mkdir(file_folder);
                end;
                fp = fopen(feature_names{j},'wb');
                fwrite(fp, [feature_dim 1 4 5], 'int32');
                fwrite(fp, feature(:,j), 'float32');
                fclose(fp);
            end;
        end;
    end;
% feature1 = features(:,1:250000);
% feature2 = features(:,250001:500000);
% feature3 = features(:,500001:750000);
% feature4 = features(:,750001:1000000);
% save('sphereface_megaface_1.mat','feature1');
% save('sphereface_megaface_2.mat','feature2');
% save('sphereface_megaface_3.mat','feature3');
% save('sphereface_megaface_4.mat','feature4');

end