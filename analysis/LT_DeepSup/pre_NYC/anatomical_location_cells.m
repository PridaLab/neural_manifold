cellType = [4., 1., 5., 0., 4., 4., 4., 4., 4., 1., 0., 1., 4., 5., 5., 4., 1., ...
       2., 3., 1., 1., 4., 4., 1., 4., 5., 4., 4., 4., 5., 5., 4., 4., 4., ...
       4., 1., 1., 1., 4., 1., 1., 4., 4., 4., 4., 5., 5., 4., 1., 1., 4., ...
       4., 1., 1., 1., 1., 4., 5., 4., 1., 1., 4., 5., 5., 5., 1., 4., 4., ...
       1., 4., 1., 4., 5., 4., 4., 4., 1., 2., 4., 4., 1., 1., 4., 1., 4., ...
       4., 4., 5., 1., 1., 4., 5., 4., 5., 4., 4., 4., 1., 1., 4., 1., 4., ...
       4., 5., 4., 1., 1., 4., 0., 0., 1., 5., 4., 4., 1., 4., 4., 1., 4., ...
       1., 4., 4., 1., 5., 4., 5., 1., 1., 4., 5., 4., 4., 4., 2., 4., 4., ...
       5., 5., 5., 4., 4., 0., 5., 4., 1., 1., 4., 4., 2., 4., 4., 4., 1., ...
       1., 4., 4., 3., 4., 1., 1., 1., 4., 4., 1., 5., 4., 1., 4., 4., 4., ...
       4., 1., 1., 5., 1., 1., 4., 1., 4., 4., 1., 4., 4., 4., 4., 1., 1., ...
       0., 0., 1., 4., 5., 4., 4.];


image_dir = '/home/julio/Documents/SP_project/LT_DeepSup/data/GC5_nvista/Inscopix_data/GC5_lt_cell_images/';
files = dir(image_dir);
files_image = files(contains({files(:).name}, 'GC5_lt_raw__C', 'IgnoreCase',true));

total_image = zeros(397,637,3);
for pyCell=1:length(cellType)
    localCell = tracesEvents.gla_guide.local_all(pyCell);

    localName = int2str(localCell);
    for ii=1:3-length(localName)
        localName = ['0', localName];
    end
    image_idx = contains({files_image(:).name}, localName, 'IgnoreCase',true);

    image_index = find(image_idx);
    image = imread([files_image(image_index).folder, ...
                        '/',files_image(image_index).name]);
    
    image(image>0) = 1;
    if cellType(pyCell)==0
        total_image(:,:,1) = total_image(:,:,1) + image;
        total_image(:,:,2) = total_image(:,:,2) + image;
        total_image(:,:,3) = total_image(:,:,3) + image;
    elseif cellType(pyCell)==1
        total_image(:,:,2) = total_image(:,:,2) + image;
    elseif cellType(pyCell) == 2
        total_image(:,:,1) = total_image(:,:,1) + image;
        total_image(:,:,2) = total_image(:,:,2) + image;
    elseif cellType(pyCell) == 3
        total_image(:,:,1) = total_image(:,:,1) + image;
        total_image(:,:,2) = total_image(:,:,2) + 0.5*image;
    elseif cellType(pyCell) == 4
        total_image(:,:,1) = total_image(:,:,1) + image;
        total_image(:,:,3) = total_image(:,:,3) + image;
    else
        total_image(:,:,1) = total_image(:,:,1) + 0.5*image;
        total_image(:,:,2) = total_image(:,:,2) + 0.5*image;
        total_image(:,:,3) = total_image(:,:,3) + 0.5*image;
    end    
end

imshow(total_image)
    contourImage = image-circshift(image,1,1) + image-circshift(image,-1,1) + image-circshift(image,1,2) + image-circshift(image,-1,2);
    contourImage(contourImage~=0) = 1;


GC5_nvista = files(contains({files(:).name}, 'GC5_lt_raw__C', 'IgnoreCase',true));
