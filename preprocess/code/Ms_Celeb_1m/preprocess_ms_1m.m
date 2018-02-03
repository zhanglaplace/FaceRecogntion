function preprocess_ms_1m()
    base64 = org.apache.commons.codec.binary.Base64;
    
    root_folder = '/home/zf/deeplearning/datasets/face-recognition/ms-1m/data';
    if exist(root_folder,'dir') == 0
        mkdir(root_folder);
    end
    tsv_file = '/home/zf/deeplearning/datasets/face-recognition/ms-1m/MsCelebV1-Faces-Aligned.tsv';
    tsv_fid = fopen(tsv_file,'r');
    
    image_id = 0;
    while ~feof(tsv_fid)
        line = fgetl(tsv_fid); 
        C = strsplit(line,'\t');
        folder = fullfile(root_folder,C{1});
        if exist(folder,'dir') == 0
            mkdir(folder);
        end
        image_id = image_id +1;
        filename = fullfile(folder,[C{2} '-' C{5} '.jpg']);
        if exist(filename,'file') ~= 0
            continue;
        end
        raw  = base64decode(C{7});
        jImg = javax.imageio.ImageIO.read(java.io.ByteArrayInputStream(raw));
        h = jImg.getHeight;
        w = jImg.getWidth;
        p = typecast(jImg.getData.getDataStorage, 'uint8');
        img = permute(reshape(p, [3 w h]), [3 2 1]);
        img = img(:,:,[3 2 1]);
        %imshow(img);
        imwrite(img,filename);
        %disp([num2str(image_id) '/' num2str(total_lines) ' ' filename]);
        %image_id = image_id + 1;
       
        if mod(image_id,1000) == 0
            fprintf('process %d images\n',image_id);
        end
        
    end
    fprintf('total images: %d',image_id);
    fclose(tsv_fid);

end


function output = base64decode(input)
%BASE64DECODE Decode Base64 string to a byte array.
%
%    output = base64decode(input)
%
% The function takes a Base64 string INPUT and returns a uint8 array
% OUTPUT. JAVA must be running to use this function. The result is always
% given as a 1-by-N array, and doesn't retrieve the original dimensions.
%
% See also base64encode

error(javachk('jvm'));
if ischar(input), input = uint8(input); end

output = typecast(org.apache.commons.codec.binary.Base64.decodeBase64(input), 'uint8')';

end