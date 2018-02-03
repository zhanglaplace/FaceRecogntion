function faceScrub_force_align()
    folder = '/data/MegaFace/FaceScrub/';
    target_folder = '/home/zf/deeplearning/datasets/face-recognition/megaface/FaceScrub-112x96/';
    load image_list_from_file.mat
    load boxes.mat
       
    
    if ~exist('image_list','var')
        actor_file = fopen('facescrub_actors_nohead.txt','r');
        actress_file = fopen('facescrub_actresses_nohead.txt','r');
        image_list = {};
        boxes = [];
        p = 1;
        while ~feof(actor_file)
            str_line = fgetl(actor_file);
            C = strsplit(str_line,'\t');
            boxes(p,:) = sscanf(C{end-1},'%d,%d,%d,%d');
            boxes(p,3) = boxes(p,3)-boxes(p,1);
            boxes(p,4) = boxes(p,4)-boxes(p,2);
            slash_pos = strfind(C{4},'/');
            slash_pos = slash_pos(end);
            dot_pos = strfind(C{4}(slash_pos:end),'.');
            tmp_file = [folder C{1} '/' C{1} '_' C{3}];
            if isempty(dot_pos)
                
                if exist(tmp_file,'file')
                    movefile(tmp_file,[tmp_file '.jpg']);
                end
                suffix = '.jpg';
            else
                dot_pos = dot_pos(end);
                dot_pos = dot_pos+slash_pos-1;
                suffix = C{4}(dot_pos:end);
                if suffix(end) == '&'
                    suffix = suffix(1:end-1);
                end
                if strcmp(suffix,'.gif') ~= 0
                    try
                        img = imread([tmp_file suffix]);
                    catch
                        continue;
                    end
                    suffix='.jpg';
                    imwrite(img,[tmp_file suffix]);
                end
            end
            image_list{p} = [tmp_file suffix];
            p =p +1;
            
        end
        
        actor_num = p-1;
        while ~feof(actress_file)
            
            str_line = fgetl(actress_file);
            C = strsplit(str_line,'\t');
            boxes(p,:) = sscanf(C{end-1},'%d,%d,%d,%d');
            boxes(p,3) = boxes(p,3)-boxes(p,1);
            boxes(p,4) = boxes(p,4)-boxes(p,2);
            slash_pos = strfind(C{4},'/');
            slash_pos = slash_pos(end);
            dot_pos = strfind(C{4}(slash_pos:end),'.');
            tmp_file = [folder C{1} '/' C{1} '_' C{3}];
            if isempty(dot_pos)
                
                if exist(tmp_file,'file')
                    movefile(tmp_file,[tmp_file '.jpg']);
                end
                suffix = '.jpg';
            else
                dot_pos = dot_pos(end);
                dot_pos = dot_pos+slash_pos-1;
                suffix = C{4}(dot_pos:end);
                if suffix(end) == '&'
                    suffix = suffix(1:end-1);
                end
                if strcmp(suffix,'.gif') ~= 0
                    try
                        img = imread([tmp_file suffix]);
                    catch
                        continue;
                    end
                    suffix='.jpg';
                    imwrite(img,[tmp_file suffix]);
                end
            end
            image_list{p} = [tmp_file suffix];
            p =p +1;
            
        end
        fclose(actor_file);
        fclose(actress_file);
       
    end
    %save boxes.mat boxes
    %save image_list_from_file.mat image_list
    
    
    %% only force this is enough
    test_list_file = '/data/MegaFace/devkit/templatelists/facescrub_uncropped_features_list.json';
    json_string = fileread(test_list_file);
    json_string = json_string(strfind(json_string,'path')+8:end);
    test_list = regexp(json_string(8:end), '"(.*?)"','tokens');
    
    
    for i=1:length(test_list)
        dot_pos = strfind(test_list{i}{1},'.');
        if isempty(dot_pos)||(dot_pos(end) ~= length(test_list{i}{1}) - 3&&dot_pos(end) ~= length(test_list{i}{1}) - 4)
           disp(test_list{i}{1});
           test_list{i}{1} = [test_list{i}{1} '.jpg'];
        end;
        if ~isempty(dot_pos) && strcmp(test_list{i}{1}(dot_pos(end):end), '.gif')
            test_list{i}{1} = [test_list{i}{1}(1:end-3) 'jpg'];
        end;
        test_list{i} = [folder test_list{i}{1}];
    end;
    %load image_list_from_file.mat
    test_list_fid = fopen('test_faceScrub_lose.txt','r');
    total_count= 410;
    mega_test_image = {};
    force = true;
    if ~force 
        for tl = 1:length(test_list)
            if exist(test_list{tl},'file') == 0
                fprintf(noface_fid,[test_list{tl} 'do not exist \n']);
                continue;
            end
            %[file_folder, ~, ~] = fileparts(test_list{tl});
            target_filename = strrep(test_list{tl},folder, target_folder);
            if exist(target_filename, 'file')
                mega_test_image{tl} = target_filename;
                continue;
            end;
            fprintf(noface_fid,[test_list{tl} ' do not get face \n']);
        end
        save megaFace_faceScrub_test_crop_list.mat mega_test_image;
    else
       for tl = 1:total_count
           image_name = fgetl(test_list_fid);
           image_name = image_name(1:end-1);
           if exist(image_name,'file') == 0
                disp([num2str(tl),'/',num2str(total_count),'!' image_name]);
                continue;
           end
           image_id = 0;
           for i=1:length(image_list)
                if strcmp(image_list{i}, image_name) 
                    image_id = i;
                    break;
                end;
           end
           if image_id == 0
               %% office image 
               target_filename = strrep(image_name,folder, target_folder);
               %if exist(target_filename,'file')
                   %continue;
               %end
               tmp_file = strrep(image_name,' ','_' );
               tmp_pos = strfind(tmp_file,'.');
               tmp_file = [tmp_file(1:tmp_pos(end)-1) '.png'];
               tmp_file = strrep(tmp_file,folder,'/home/zf/deeplearning/datasets/face-recognition/megaface/facescrub_aligned/');
               if exist(tmp_file,'file')
                    try
                        img = imread(tmp_file);
                    catch
                       disp([num2str(tl),'/',num2str(total_count),'!!!!' image_name ]);
                       continue;
                    end
                    if isa(img,'uint16')
                        img = uint8(img / 256);
                    end;
                    if size(img, 3) < 3
                       img(:,:,2) = img(:,:,1);
                       img(:,:,3) = img(:,:,1);
                    end
                     img = imresize(img,[112,96]);
                     imwrite(img,target_filename);
                     %copyfile(tmp_file,target_filename);
               else
                   disp([num2str(tl),'/',num2str(total_count),'!!' image_name ]);
               end
           else
               target_filename = strrep(image_list{image_id},folder, target_folder);
              % if exist(target_filename, 'file')
                  %  continue;
              % end
               
               try
                    img = imread(image_name);
               catch
                   disp([num2str(tl),'/',num2str(total_count),'!!!!' image_name ]);
                   continue;
               end
               if isa(img,'uint16')
                    img = uint8(img / 256);
                end;
                if size(img, 3) < 3
                   img(:,:,2) = img(:,:,1);
                   img(:,:,3) = img(:,:,1);
                end
                img = img(boxes(image_id,2):boxes(image_id,2)+boxes(image_id,4),boxes(image_id,1):boxes(image_id,1)+boxes(image_id,3));
                img = imresize(img,[112,96]);
                imwrite(img,target_filename);            
           end
       
        end;
 
    end
    fclose(test_list_fid);
    
    
    
    
    
    
    
    
end