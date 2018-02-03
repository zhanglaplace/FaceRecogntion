folder = '/home/zf/deeplearning/datasets/face-recognition/youtube/YouTubeFaces/aligned_images_DB';
%addpath('..');
%image_list = get_image_list_in_folder(folder);
%save image_list_linux.mat image_list
load image_list_linux.mat
target_folder = '/home/zf/deeplearning/datasets/face-recognition/youtube/YouTubeFaces/aligned_images_DB-112X96';
target_folder_guoke = '/home/zf/deeplearning/datasets/face-recognition/youtube/YouTubeFaces/aligned_images_DB-112X112';

if exist(target_folder, 'dir')==0
    mkdir(target_folder);
end;

if exist(target_folder_guoke, 'dir')==0
    mkdir(target_folder_guoke);
end;


caffe_path = '/home/zf/deeplearning/caffe/matlab';
addpath(caffe_path);

pdollar_toolbox_path='/home/zf/deeplearning/caffe/face_example/face-recognition/Sphereface/tools/toolbox';
addpath(genpath(pdollar_toolbox_path));

MTCNN_path = '/home/zf/deeplearning/caffe/face_example/face-recognition/Sphereface/tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1';
caffe_model_path=[MTCNN_path , '/model'];
addpath(genpath(MTCNN_path));

coord5points = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299; ...
                51.6963, 51.5014, 71.7366, 92.3655, 92.2041];
imgSize = [112, 96];

coord5points_guoke = [38.2946, 73.5318, 56.0252, 41.5493, 70.7299; ...
                51.6963, 51.5014, 71.7366, 92.3655, 92.2041];
imgSize_guoke = [112,112];            
            
align_method = 'yandong';% wuxiang or yandong
            
%caffe.set_mode_cpu();
gpu_id=0;
caffe.set_mode_gpu();	
caffe.set_device(gpu_id);
caffe.reset_all();

%three steps's threshold
%threshold=[0.6 0.7 0.7]
threshold=[0.6 0.7 0.9];
minsize = 40;%80;

%scale factor
factor=0.85;%0.709;

%load caffe models
PNet = caffe.Net('/home/zf/deeplearning/caffe/face_example/face-recognition/Sphereface/tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model/det1.prototxt','/home/zf/deeplearning/caffe/face_example/face-recognition/Sphereface/tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model/det1.caffemodel', 'test');
RNet = caffe.Net('/home/zf/deeplearning/caffe/face_example/face-recognition/Sphereface/tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model/det2.prototxt', ...
                 '/home/zf/deeplearning/caffe/face_example/face-recognition/Sphereface/tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model/det2.caffemodel', 'test');
ONet = caffe.Net('/home/zf/deeplearning/caffe/face_example/face-recognition/Sphereface/tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model/det3.prototxt', ...
                 '/home/zf/deeplearning/caffe/face_example/face-recognition/Sphereface/tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model/det3.caffemodel', 'test');
faces=cell(0);	

align_list_112x96  = {};
align_list_112x112={};
for image_id = 1:length(image_list);
    img = imread(image_list{image_id});
    if size(img, 3) < 3
       img(:,:,2) = img(:,:,1);
       img(:,:,3) = img(:,:,1);
    end
    [file_folder, file_name, file_ext] = fileparts(image_list{image_id});
    target_filename = strrep(image_list{image_id},folder, target_folder);
    target_filename_guoke = strrep(image_list{image_id},folder, target_folder_guoke);
    assert(strcmp(target_filename, image_list{image_id})==0);
    assert(strcmp(target_filename_guoke, image_list{image_id})==0);
    [file_folder, file_name, file_ext] = fileparts(target_filename);
    if exist(file_folder,'dir')==0
        mkdir(file_folder);
    end;
    [file_folder_guoke, file_name, file_ext] = fileparts(target_filename_guoke);
    if exist(file_folder_guoke,'dir')==0
        mkdir(file_folder_guoke);
    end;
    disp([num2str(image_id) '/' num2str(length(image_list)) ' ' target_filename]);
    disp([num2str(image_id) '/' num2str(length(image_list)) ' ' target_filename_guoke]);
    
    if exist(target_filename,'file') ~= 0
        align_list_112x96 = [align_list_112x96;target_filename];
        align_list_112x112 = [align_list_112x112;target_filename_guoke];
        continue;
    end
    
    [boundingboxes, points] = detect_face(img, minsize, PNet, RNet, ONet, threshold, false, factor);

    if isempty(boundingboxes)
        continue;
    end;
    default_face = 1;
    if size(boundingboxes,1) > 1
        for bb=2:size(boundingboxes,1)
            if abs((boundingboxes(bb,1) + boundingboxes(bb,3))/2 - size(img,2) / 2) + abs((boundingboxes(bb,2) + boundingboxes(bb,4))/2 - size(img,1) / 2) < ...
                    abs((boundingboxes(default_face,1) + boundingboxes(default_face,3))/2 - size(img,2) / 2) + abs((boundingboxes(default_face,2) + boundingboxes(default_face,4))/2 - size(img,1) / 2)
                default_face = bb;
            end;
        end;
    end;
    facial5points = double(reshape(points(:,default_face),[5 2])');
    if strcmp(align_method, 'wuxiang') > 0
        [res, eyec2, cropImg, resize_scale] = align_face_WX(img,facial5points',144,48,48);
        cropImg = uint8(cropImg);
    else
        Tfm =  cp2tform(facial5points', coord5points', 'similarity');
        cropImg = imtransform(img, Tfm, 'XData', [1 imgSize(2)],...
                                      'YData', [1 imgSize(1)], 'Size', imgSize);
        Tfm_guoke =  cp2tform(facial5points', coord5points_guoke', 'similarity');
        cropImg_guoke = imtransform(img, Tfm_guoke, 'XData', [1 imgSize_guoke(2)],...
                                      'YData', [1 imgSize_guoke(1)], 'Size', imgSize_guoke);
    end;
    imwrite(cropImg, target_filename);
    imwrite(cropImg_guoke, target_filename_guoke);
	% show detection result
% 	numbox=size(boundingboxes,1);
%     figure(1);
% 	imshow(img)
% 	hold on; 
% 	for j=1:numbox
% 		plot(points(1:5,j),points(6:10,j),'g.','MarkerSize',10);
% 		r=rectangle('Position',[boundingboxes(j,1:2) boundingboxes(j,3:4)-boundingboxes(j,1:2)],'Edgecolor','g','LineWidth',3);
%     end;
%     hold off;
%     figure(2);
%     imshow(cropImg);
% 	pause
end;
save align_list_112x96.mat align_list_112x96
save align_list_112x112.mat align_list_112x112
