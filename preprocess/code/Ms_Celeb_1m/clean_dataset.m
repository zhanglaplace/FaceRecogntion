function clean_dataset()

    %image_list = '/home/zf/deeplearning/datasets/face-recognition/ms-1m/data_image_list.txt';
    %image_list_fid = fopen(image_list,'r');
    root_folder = '/home/zf/deeplearning/datasets/face-recognition/ms-1m/data/';
    target_folder = '/data/ms-1m';
    clean_list = '/home/zf/deeplearning/datasets/face-recognition/ms-1m/MS-Celeb-1M_clean_list_with_id.txt';
    clean_list_fid = fopen(clean_list,'r');
    
    Error_list = '/home/zf/deeplearning/datasets/face-recognition/ms-1m/ERROR.txt';
    error_fid = fopen(Error_list,'w');

    if exist(target_folder,'dir') == 0
        mkdir(target_folder);
    end
    
    image_id = 0;
    while ~feof(clean_list_fid)
        line = fgetl(clean_list_fid);
        C = strsplit(line,'/');
        folder_C = strsplit(C{1},',');
        folder = folder_C{1};
        filename = C{2};
        image_filename = fullfile(root_folder,folder,'/',filename);
        
        image_id = image_id +1;
        
        if mod(image_id,1000) == 0
            disp(['!!!!!process ',num2str(image_id),'  images']);
        end
        if exist(image_filename,'file') == 0
            fprintf(error_fid,[image_filename,' dont exist \n']);
            continue;
        end
        target_filename = fullfile(target_folder,folder,'/',filename);
        if exist(target_filename,'file') ~= 0
            continue;
        end
        [t_folder,~,~] = fileparts(target_filename);
        disp([target_filename]);
        if exist(t_folder,'dir') == 0
            mkdir(t_folder);
        end
        try
            img = imread(image_filename);
        catch
            fprintf(error_fid,[image_filename,' read error \n']);
            continue;
        end
        img = imresize(img,[112,96]);
        imwrite(img,target_filename);
        
    end
    fclose(clean_list_fid);
    fclose(error_fid);
end