function megaFace_force_align()
    clc,clear;
    folder = '/data/MegaFace/FlickrFinal2';
    load image_list_linxu_100M.mat
    if ~exist('image_list','var')
        list_file = '/data/MegaFace/devkit/templatelists/megaface_features_list.json_1000000_1';
        json_string = fileread(list_file);
        image_list = regexp(json_string(8:end), '"(.*?)"','tokens');
        for i=1:length(image_list)
            image_list{i} = [folder '/' image_list{i}{1}];
        end;
    end;
    target_folder = '/home/zf/deeplearning/datasets/face-recognition/megaface/MegaFace-112x96';
    if exist(target_folder, 'dir')==0
        mkdir(target_folder);
    end;
    image_list_len = length(image_list);
    megaface_test_crop_list_100w= {};
    error_list_fid = fopen('megaface_100w_error.txt','w');
    be_process_fid = fopen('megaface_100w_last_preprocess.txt','w');
    for image_id = 1:image_list_len
        target_filename = strrep(image_list{image_id},folder, target_folder);
        megaface_test_crop_list_100w{image_id} = target_filename;
        if exist(target_filename,'file')
            continue;
        end
        if ~exist([image_list{image_id}, '.json'],'file')
            sprintf(error_list_fid,[image_list{image_id} ', json not exist\n']);
            continue ;
        end;
        try
            img = imread(image_list{image_id});
        catch
            sprintf(error_list_fid,[image_list{image_id} ', read  error\n']);
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

        json = parse_json(fileread([image_list{image_id} '.json']));

        if isfield(json{1}, 'bounding_box')
            bboxes = json{1}.bounding_box;
            img = img(max(1,int16(bboxes.y+1)):min(size(img,1),int16(bboxes.y+1+bboxes.height)),max(1,int16(bboxes.x+1)):min(size(img,2),int16(bboxes.x+1+bboxes.width)));
            %img = img(int8(bboxes.x+1:bboxes.x+1+bboxes.width),int8(bboxes.y+1:bboxes.y+1+bboxes.height));
            cropImg = imresize(img,[112,96]);
            imwrite(cropImg,target_filename);
            %imshow(cropImg);
            disp([num2str(image_id),'/' num2str(image_list_len) ' ' target_filename]);
        else
            sprintf(error_list_fid,[image_list{image_id} ', not has bboundingbox \n']);
        end;
       
    end
    fclose(error_list_fid);
    fclose(be_process_fid);
    save megaface_test_crop_list_100w.mat megaface_test_crop_list_100w
end