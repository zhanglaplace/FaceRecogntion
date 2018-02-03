function preprocess_faceScrub()
%% align FaceScrub
root_folder = '/data/MegaFace/FaceScrub';
%image_list = get_image_list_in_folder(root_folder);
%save image_list_linux_faceScrub.mat image_list ;
load image_list_linux_faceScrub.mat
target_folder = '/data/MegaFace/FaceScrub-112x96';
if exist(target_folder, 'dir')==0
    mkdir(target_folder);
end;

%% add env params
pdollar_toolbox_path='/home/zf/deeplearning/caffe/face_example/face-recognition/Sphereface/tools/toolbox/';
addpath(genpath(pdollar_toolbox_path));
MTCNN_path = '/home/zf/deeplearning/caffe/face_example/face-recognition/Sphereface/tools/MTCNN_face_detection_alignment/code/codes/MTCNNv2';
caffe_model_path=[MTCNN_path , '/model/'];
addpath(MTCNN_path);
caffe_path = '/home/zf/deeplearning/caffe/matlab';
addpath(caffe_path);

gpu_id=0;
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);

%% mtcnn setting
coord5points = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299; ...
                51.6963, 51.5014, 71.7366, 92.3655, 92.2041];
imgSize = [112, 96];

%three steps's threshold
threshold=[0.6 0.7 0.7];
%MatMTCNN('set_threshold', threshold);
minsize = 100;
%scale factor
factor=0.709;

%load caffe models
prototxt_dir =strcat(caffe_model_path,'/det1.prototxt');
model_dir = strcat(caffe_model_path,'/det1.caffemodel');
PNet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir = strcat(caffe_model_path,'/det2.prototxt');
model_dir = strcat(caffe_model_path,'/det2.caffemodel');
RNet=caffe.Net(prototxt_dir,model_dir,'test');	
prototxt_dir = strcat(caffe_model_path,'/det3.prototxt');
model_dir = strcat(caffe_model_path,'/det3.caffemodel');
ONet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir =  strcat(caffe_model_path,'/det4.prototxt');
model_dir =  strcat(caffe_model_path,'/det4.caffemodel');
LNet=caffe.Net(prototxt_dir,model_dir,'test');
image_list_len = length(image_list);

%% record no face
noface_fid = fopen('noface.txt','w');

for image_id = 1:image_list_len
    clear cropImg;
    target_filename = strrep(image_list{image_id},root_folder, target_folder);
    if exist(target_filename, 'file')
        %disp([num2str(image_id) '/' num2str(image_list_len) image_list{image_id}]);
        continue;
    end;
    try
        img = imread(image_list{image_id});
    catch
        fprintf(noface_fid,[image_list{image_id} ',read error\n']);
        continue;
    end;
    
    if size(img, 3) < 3
       img(:,:,2) = img(:,:,1);
       img(:,:,3) = img(:,:,1);
    end
    
    
    assert(strcmp(target_filename, image_list{image_id})==0);
    [file_folder, ~, ~] = fileparts(target_filename);
    if exist(file_folder,'dir')==0
        mkdir(file_folder);
    end;
       
    [boundingboxes,points]=detect_face(img,min([minsize size(img,1)/2 size(img,2)/2]),PNet,RNet,ONet,LNet,threshold,false,factor);
    default_face = 1;
    if ~isempty(boundingboxes)
        for bb =2:size(boundingboxes,1)
            if abs((boundingboxes(bb,1) + boundingboxes(bb,3))/2 - size(img,2) / 2) + abs((boundingboxes(bb,2) + boundingboxes(bb,4))/2 - size(img,1) / 2) < ...
                abs((boundingboxes(default_face,1) + boundingboxes(default_face,3))/2 - size(img,2) / 2) + abs((boundingboxes(default_face,2) + boundingboxes(default_face,4))/2 - size(img,1) / 2)
                default_face = bb;
            end;
        end;
    facial5points = double(reshape(points(:,default_face),[5 2])');
                
     %facial5points = [points(default_face,1:2:9);points(default_face,2:2:10)];
    Tfm =  cp2tform(facial5points', coord5points', 'similarity');
    cropImg = imtransform(img, Tfm, 'XData', [1 imgSize(2)],...
                                  'YData', [1 imgSize(1)], 'Size', imgSize);
    fprintf('%d/%d, %s [%d,%d]\n',image_id,image_list_len,target_filename,int32(boundingboxes(default_face,3)),int32(boundingboxes(default_face,4)));
    else
        fprintf(noface_fid,[image_list{image_id} '\n']);
        continue;
    end;  
    if exist('cropImg','var')
        imwrite(cropImg, target_filename);
    end;
end;

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


