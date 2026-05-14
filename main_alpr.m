%% ========================================================================
% Zagazig University — Faculty of Engineering
% Electronics and Communications Engineering Department
% Course: ECE 228: Image Processing
% Project: Automatic License Plate Recognition (Classical Image Processing)
%
% Group Number: 6
% Team Members:
% 1. Ahmed Mohamed Attia Mohamed
% 2. Ali El-Shawadfy Abdallah El-Sayed
% 3. Omar Hosny Mohamed Ahmed Abouzeid
% 4. Anas Ali Hammad El-Sayed
% 5. Abdelhay Lotfy El-Sayed El-Gawahry
%% ========================================================================

clc; clear; close all;

%% 1. Paths Setup
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd; 
end

dataset_dir = fullfile(script_dir, '..', 'dataset');
if ~exist(dataset_dir,'dir') || isempty(dir(fullfile(dataset_dir,'*.jpg')))
    disp('Select your dataset folder ...');
    dataset_dir = uigetdir(script_dir, 'Select dataset folder (.jpg files)');
    if dataset_dir == 0
        error('Cancelled.'); 
    end
end

results_dir = fullfile(dataset_dir, 'results');
if ~exist(results_dir,'dir')
    mkdir(results_dir); 
end

%% 2. Load Templates
templates_dir = fullfile(results_dir, 'templates');
templates = load_real_templates(templates_dir);
fprintf('Templates loaded: %d\n', length(templates));

%% Deduplicate Files
all_files     = [dir(fullfile(dataset_dir,'*.jpg')); ...
                 dir(fullfile(dataset_dir,'*.JPG'))];
[~, keep_idx] = unique(lower({all_files.name}));
img_files     = all_files(sort(keep_idx));
num_images    = length(img_files);
fprintf('Found %d images in: %s\n', num_images, dataset_dir);

%% 3. Accumulators Setup
total_detected    = 0;
results_filenames = cell(num_images,1);
results_detected  = false(num_images,1);
results_letters   = cell(num_images,1); 
results_numbers   = cell(num_images,1); 

%% 4. Main Processing Loop
for img_idx = 1:num_images

    filename = img_files(img_idx).name;
    filepath = fullfile(dataset_dir, filename);
    img_orig = imread(filepath);
    
    results_filenames{img_idx} = filename;

    [~, orig_w, ~] = size(img_orig);
    img = imresize(img_orig, 800/orig_w);
    [img_h, img_w, ~] = size(img);

    if size(img,3) == 3
        gray_img = rgb2gray(img);
        hsv_img  = rgb2hsv(img);
    else
        gray_img = img;
        hsv_img  = repmat(double(img)/255,[1 1 3]);
    end
    filtered_img = medfilt2(gray_img,[3 3]);

    % Color Masks
    hue = hsv_img(:,:,1); 
    sat = hsv_img(:,:,2); 
    val = hsv_img(:,:,3);

    blue_mask  = (hue>0.52 & hue<0.75) & (sat>0.25) & (val>0.12);
    blue_mask  = imclose(blue_mask, strel('rectangle',[3 8]));
    blue_mask  = bwareaopen(blue_mask, 50);

    white_mask = (sat<0.20) & (val>0.60);
    white_mask = imclose(white_mask, strel('rectangle',[3 8]));
    white_mask = bwareaopen(white_mask,100);

    % Edge Map Generation
    [~,tv] = edge(filtered_img,'sobel','vertical');
    edge_v  = edge(filtered_img,'sobel','vertical',  tv*0.7);
    [~,th] = edge(filtered_img,'sobel','horizontal');
    edge_h  = edge(filtered_img,'sobel','horizontal', th*0.7);

    edge_combined = bwareaopen(edge_v,10);
    dilated  = imdilate(edge_combined, strel('rectangle',[3 15]));
    closed   = imclose(dilated,        strel('rectangle',[4  4]));
    filled   = imfill(closed,'holes');

    max_area  = img_h * img_w * 0.05;
    clean_img = bwareaopen(filled,300);
    labeled   = bwlabel(clean_img);
    sp        = regionprops(labeled,'Area','PixelIdxList');
    for s=1:length(sp)
        if sp(s).Area > max_area
            clean_img(sp(s).PixelIdxList) = 0; 
        end
    end

    % Candidate Scoring
    stats      = regionprops(clean_img,'BoundingBox','Area','Extent');
    blue_stats = regionprops(blue_mask,'BoundingBox','Area');

    plate_detected = false;
    best_bbox      = [];
    best_score     = -Inf;

    for i=1:length(stats)
        bbox=stats(i).BoundingBox; 
        w=bbox(3); 
        h=bbox(4);
        ar=w/h; 
        area=stats(i).Area; 
        ext=stats(i).Extent;
        yc=bbox(2)+h/2; 
        xc=bbox(1)+w/2;

        if ar<1.5 || ar>6.0, continue; end
        if area<300 || area>max_area, continue; end
        if w<img_w*0.03 || w>img_w*0.35, continue; end
        if h<img_h*0.01 || h>img_h*0.18, continue; end
        if ext<0.35, continue; end
        if yc<img_h*0.15, continue; end

        ec=imcrop(edge_v,bbox); 
        vd=sum(ec(:))/(w*h);
        if vd<0.03, continue; end
        score = vd * 100;

        ehc=imcrop(edge_h,bbox); 
        hd=sum(ehc(:))/(w*h);
        if hd>0
            vhr = vd/(hd+0.001);
            if vhr>0.8
                score = score + min(vhr,5)*2; 
            end
        end

        for b=1:length(blue_stats)
            bb2=blue_stats(b).BoundingBox;
            bxc=bb2(1)+bb2(3)/2; 
            byb=bb2(2)+bb2(4);
            if abs(bxc-xc)<w*0.6 && byb>=bbox(2)-h*0.5 && byb<=bbox(2)+h*0.6
                score = score + 15; 
                break;
            end
        end

        bc=imcrop(blue_mask,bbox); 
        br=sum(bc(:))/(w*h);
        if br>0.05 && br<0.5
            score = score + br*30; 
        end

        wc=imcrop(white_mask,bbox); 
        wr=sum(wc(:))/(w*h);
        if wr>0.15 && wr<0.85
            score = score + wr*10; 
        end

        yn = yc/img_h;
        if yn>0.4 && yn<0.85, score = score + 5; end
        if ar>2.0 && ar<5.0,  score = score + 3; end

        gc=imcrop(filtered_img,bbox);
        if ~isempty(gc)
            score = score + min(std(double(mean(gc,1)))/10,5); 
        end

        if score > best_score
            best_score = score; 
            best_bbox = bbox; 
            plate_detected = true;
        end
    end

    % Fallback: Blue-band expansion
    if ~plate_detected
        for b=1:length(blue_stats)
            bb2=blue_stats(b).BoundingBox;
            bw2=bb2(3); 
            bh2=bb2(4);
            if bw2/bh2>2.0 && bw2>img_w*0.03 && blue_stats(b).Area>80
                exp_bb=[bb2(1)-bw2*0.05, bb2(2)-bh2*0.2, bw2*1.1, bh2*4.5];
                exp_bb(1) = max(1,exp_bb(1)); 
                exp_bb(2) = max(1,exp_bb(2));
                if exp_bb(1)+exp_bb(3) > img_w, exp_bb(3) = img_w-exp_bb(1); end
                if exp_bb(2)+exp_bb(4) > img_h, exp_bb(4) = img_h-exp_bb(2); end
                
                ec2 = imcrop(edge_v,exp_bb);
                d2 = sum(ec2(:))/(exp_bb(3)*exp_bb(4));
                if d2 > 0.02
                    best_bbox = exp_bb; 
                    best_score = d2*50;
                    plate_detected = true; 
                    break;
                end
            end
        end
    end

    if plate_detected && best_score<3
        plate_detected=false; 
        best_bbox=[]; 
    end
    results_detected(img_idx) = plate_detected;

    % Plate Cropping & Recognition
    recognized_string = '';
    str_num = '';
    str_let = '';
    bottom_plate = []; 
    plate_bw = []; 
    valid_numbers_boxes = []; 
    valid_letters_boxes = [];

    if plate_detected
        total_detected = total_detected + 1;

        pad = 5;
        pb = [max(1, best_bbox(1)-pad), max(1, best_bbox(2)-pad), ...
              min(best_bbox(3)+2*pad, img_w-best_bbox(1)+pad), ...
              min(best_bbox(4)+2*pad, img_h-best_bbox(2)+pad)];

        full_plate = imcrop(gray_img, pb);
        bottom_plate = full_plate; 
        
        plate_bw  = robust_binarize(bottom_plate);
        raw_chars = segment_characters(plate_bw);

        if ~isempty(raw_chars)
            raw_chars = sortrows(raw_chars, 1, 'ascend');  
            
            plate_width = size(plate_bw, 2);
            separator_x = plate_width * 0.48; 
            
            numbers_boxes = [];
            letters_boxes = [];
            
            for g = 1:size(raw_chars,1)
                char_center_x = raw_chars(g, 1) + (raw_chars(g, 3)/2);
                if char_center_x < separator_x
                    numbers_boxes = [numbers_boxes; raw_chars(g,:)];
                else
                    letters_boxes = [letters_boxes; raw_chars(g,:)];
                end
            end
            
            export_dir = fullfile(results_dir, 'extracted_chars');
            if ~exist(export_dir, 'dir'), mkdir(export_dir); end
            
            % Validate Numbers
            matched_numbers = {};
            num_scores = [];
            for c=1:size(numbers_boxes,1)
                char_img  = imcrop(plate_bw, numbers_boxes(c,:));
                char_img  = imresize(char_img,[42 24]);
                
                if ~isempty(templates)
                    [best_match, score] = match_character(char_img, templates, true);
                    if score > 0.2
                        matched_numbers{end+1} = best_match;
                        num_scores(end+1) = score;
                        valid_numbers_boxes = [valid_numbers_boxes; numbers_boxes(c,:)];
                        imwrite(char_img, fullfile(export_dir, sprintf('img_%02d_num_%d.png', img_idx, c)));
                    end
                end
            end
            
            if length(matched_numbers) > 4
                [~, sort_idx] = sort(num_scores, 'descend');
                top_idx = sort(sort_idx(1:4)); 
                matched_numbers = matched_numbers(top_idx);
                valid_numbers_boxes = valid_numbers_boxes(top_idx, :);
            end
            
            % Validate Letters
            matched_letters = {};
            let_scores = [];
            for c=1:size(letters_boxes,1)
                char_img  = imcrop(plate_bw, letters_boxes(c,:));
                char_img  = imresize(char_img,[42 24]);
                
                if ~isempty(templates)
                    [best_match, score] = match_character(char_img, templates, false);
                    if score > 0.2
                        matched_letters{end+1} = best_match;
                        let_scores(end+1) = score;
                        valid_letters_boxes = [valid_letters_boxes; letters_boxes(c,:)];
                        imwrite(char_img, fullfile(export_dir, sprintf('img_%02d_let_%d.png', img_idx, c)));
                    end
                end
            end
            
            if length(matched_letters) > 3
                [~, sort_idx] = sort(let_scores, 'descend');
                top_idx = sort(sort_idx(1:3)); 
                matched_letters = matched_letters(top_idx);
                valid_letters_boxes = valid_letters_boxes(top_idx, :);
            end
            
            % Final String Formatting
            str_num = strjoin(flip(matched_numbers), ' '); 
            str_let = strjoin(flip(matched_letters), ' '); 
            
            recognized_string = strtrim([str_let, '   ', str_num]);
        end

        if isempty(recognized_string)
            recognized_string = ''; 
            str_num = '';
            str_let = '';
        end
        detection_status = 'detected';
    else
        recognized_string = '';
        str_num = '';
        str_let = '';
        detection_status = 'not detected';
    end

    % Update Data Accumulators for Excel
    results_letters{img_idx} = str_let;
    results_numbers{img_idx} = str_num;

    % Console Output
    fprintf('\n[%02d/%02d] Processing File : %s\n', img_idx, num_images, filename);
    fprintf('Detection result        : %s\n', detection_status);
    
    if strcmp(detection_status, 'detected')
        fprintf('Letters: %s | Numbers: %s\n', str_let, str_num);
    end
    fprintf('--------------------------------------------------\n');

    % Visualisation
    if img_idx <= 100
        fh = figure('Name',sprintf('ALPR - %s',filename),'Position',[50 50 1400 800], 'Visible', 'off');

        subplot(2,4,1); imshow(img);          title('1. Standardized');
        subplot(2,4,2); imshow(filtered_img); title('2. Filtered Gray');
        subplot(2,4,3); imshow(blue_mask);    title('3. Blue Mask');
        subplot(2,4,4); imshow(clean_img);    title('4. Edge Candidates');

        subplot(2,4,5); imshow(img); hold on;
        if plate_detected
            rectangle('Position',best_bbox,'EdgeColor','g','LineWidth',3);
            title(sprintf('5. Detected (%.1f)',best_score));

            subplot(2,4,6);
            if ~isempty(bottom_plate), imshow(bottom_plate); end
            title('6. Plate Bottom');

            subplot(2,4,7);
            if ~isempty(plate_bw)
                imshow(plate_bw); hold on;
                if ~isempty(valid_numbers_boxes)
                    for c=1:size(valid_numbers_boxes,1)
                        rectangle('Position',valid_numbers_boxes(c,:),'EdgeColor','r','LineWidth',1.5);
                    end
                end
                if ~isempty(valid_letters_boxes)
                    for c=1:size(valid_letters_boxes,1)
                        rectangle('Position',valid_letters_boxes(c,:),'EdgeColor','b','LineWidth',1.5);
                    end
                end
            end
            total_valid = size(valid_numbers_boxes,1) + size(valid_letters_boxes,1);
            title(sprintf('7. Validated (%d chars)', total_valid));
        else
            title('5. NOT Detected');
        end

% Explicit Result Output on Image (Subplot 8) - MAXIMUM SPACE DESIGN
        subplot(2,4,8); 
        cla; 
        axis off;
        
        % CRITICAL: Fix axis limits to prevent auto-stretching
        xlim([0 1]); 
        ylim([0 1]);
        title('8. Final Result');
        
        if strcmp(detection_status, 'detected')
            % Status Text
            text(0.5, 0.85, 'Result: DETECTED', 'FontSize', 18, 'Units', 'data', ...
                 'FontName', 'Arial', 'FontWeight', 'bold', 'Color', [0 0.5 0], ...
                 'HorizontalAlignment', 'center');
             
            % Plate Background Rectangle (MAXIMUM width from 0 to 1)
            rectangle('Position', [0.0, 0.30, 1.0, 0.35], 'Curvature', 0.1, ...
                      'FaceColor', [0.95 0.95 0.95], 'EdgeColor', 'k', 'LineWidth', 2);
            
            % Blue Top Band (EGYPT)
            rectangle('Position', [0.0, 0.50, 1.0, 0.15], ...
                      'FaceColor', [0.1 0.4 0.8], 'EdgeColor', 'none');
            text(0.5, 0.575, 'EGYPT                     مصر', 'FontSize', 12, 'Units', 'data', ...
                 'FontName', 'Arial', 'FontWeight', 'bold', 'Color', 'w', ...
                 'HorizontalAlignment', 'center');
             
            % Separator Line (At 0.54 to balance 4 numbers vs 3 letters perfectly)
            line([0.54 0.54], [0.30 0.50], 'Color', 'k', 'LineWidth', 2);
             
            % Numbers (Left Side - Red) - Font reduced to 26 for more breathing room
            text(0.27, 0.40, str_num, 'FontSize', 26, 'Units', 'data', ...
                 'FontName', 'Arial', 'FontWeight', 'bold', 'Color', [0.8 0 0], ...
                 'HorizontalAlignment', 'center', 'Interpreter', 'none');
                 
            % Letters (Right Side - Blue) - Font reduced to 26 for more breathing room
            text(0.77, 0.40, str_let, 'FontSize', 26, 'Units', 'data', ...
                 'FontName', 'Arial', 'FontWeight', 'bold', 'Color', [0 0 0.8], ...
                 'HorizontalAlignment', 'center', 'Interpreter', 'none');
        else
            % Print Not Detected
            text(0.5, 0.5, 'NOT DETECTED', 'FontSize', 22, 'Units', 'data', ...
                 'FontName', 'Arial', 'FontWeight', 'bold', 'Color', [0.8 0 0], ...
                 'HorizontalAlignment', 'center');
        end

        try
            saveas(fh, fullfile(results_dir, sprintf('result_%s.png', filename(1:end-4))));
        catch
        end
        close(fh);
    end
end

%% 5. Export Results
fprintf('\n=== Done === Detected: %d / %d\n', total_detected, num_images);

T = table(results_filenames, results_detected, results_letters, results_numbers, ...
          'VariableNames',{'Filename','Is_Detected','Letters','Numbers'});

output_file = fullfile(results_dir, 'alpr_results.xlsx');

% Added try-catch to prevent crash if Excel is open
try
    writetable(T, output_file);
    fprintf('Saved: %s\n', output_file);
catch
    fprintf('\n[WARNING] Could not overwrite %s (File might be open in Excel).\n', output_file);
    alt_file = fullfile(results_dir, sprintf('alpr_results_%s.xlsx', datestr(now, 'HHMMSS')));
    writetable(T, alt_file);
    fprintf('Saved instead to: %s\n', alt_file);
end

%% LOCAL FUNCTIONS

% Evaluates target character image against loaded templates utilizing HOG features
function [best_match, best_score] = match_character(char_img, templates, is_number_mode)
    arabic_digits = {'٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
    best_match = '';
    best_score = -Inf;
    
    unique_chars = unique({templates.char});
    for uc = 1:length(unique_chars)
        current_char = unique_chars{uc};
        
        is_digit_template = ismember(current_char, arabic_digits);
        if (is_number_mode && ~is_digit_template) || (~is_number_mode && is_digit_template)
            continue; 
        end
        
        char_corrs = [];
        for k = 1:length(templates)
            if strcmp(templates(k).char, current_char)
                features_char = extractHOGFeatures(char_img, 'CellSize', [8 8]);
                feat_tpl = extractHOGFeatures(double(templates(k).img), 'CellSize', [8 8]);
                
                dist = norm(features_char - feat_tpl);
                cv = 1 / (1 + dist); 
                char_corrs = [char_corrs, cv];
            end
        end
        
        if ~isempty(char_corrs)
            class_score = mean(char_corrs);
            if class_score > best_score
                best_score = class_score;
                best_match = current_char;
            end
        end
    end
end

% Scans directory for character images and maps them to predefined Arabic representations
function templates = load_real_templates(tpl_dir)
    templates = struct('char',{},'img',{});
    keys = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ...
            'a', 'b', 't', 'th', 'g', 'h', 'kh', 'd', 'z', 'r', 's', 'sh', ...
            'sd', 'dd', 'ta', 'za', 'e', 'gh', 'f', 'q', 'k', 'l', 'm', 'n', 'he', 'w', 'y'};
    vals = {'٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩', ...
            'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'س', 'ش', ...
            'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ى'};
    map = containers.Map(keys, vals);
    
    if ~exist(tpl_dir, 'dir')
        mkdir(tpl_dir); 
        return; 
    end
    
    files = dir(fullfile(tpl_dir, '*.png'));
    idx = 1;
    for i = 1:length(files)
        fname = files(i).name;
        filepath = fullfile(tpl_dir, fname);
        
        parts = split(fname, {'-', '.'});
        prefix = lower(parts{1}); 
        if startsWith(prefix, 'tpl_'), continue; end
        
        if isKey(map, prefix)
            mapped_char = map(prefix); 
        else
            mapped_char = parts{1}; 
        end
        
        img = imread(filepath);
        if islogical(img)
            bw = img;
        else
            if size(img,3) == 3, img = rgb2gray(img); end
            bw = imbinarize(img); 
        end
        
        st = regionprops(bw, 'BoundingBox');
        if ~isempty(st), bw = imcrop(bw, st(1).BoundingBox); end
        
        templates(idx).char = mapped_char;
        templates(idx).img = logical(imresize(bw, [42 24]));
        idx = idx + 1;
    end
end

% Binarizes plate region utilizing dynamic thresholds and foreground polarity adjustments
function bw = robust_binarize(gray_plate)
    if isempty(gray_plate)
        bw=[]; 
        return; 
    end
    
    [h, w] = size(gray_plate);
    if h < 16 || w < 16
        enh = imadjust(gray_plate); 
    else
        enh = adapthisteq(gray_plate, 'ClipLimit', 0.02);
        enh = imadjust(enh);
    end

    bw1a = ~imbinarize(enh, graythresh(enh));
    bw2a = ~imbinarize(enh,'adaptive','ForegroundPolarity','dark','Sensitivity',0.45);
    bw3a = enh < 120;
    bwA  = remove_large_blobs(bwareaopen(uint8(bw1a)+uint8(bw2a)+uint8(bw3a) >= 2, 12));

    bw1b = imbinarize(enh, graythresh(enh));
    bw2b = imbinarize(enh,'adaptive','ForegroundPolarity','bright','Sensitivity',0.45);
    bw3b = enh > 135;
    bwB  = remove_large_blobs(bwareaopen(uint8(bw1b)+uint8(bw2b)+uint8(bw3b) >= 2, 12));

    if count_char_blobs(bwB,h) > count_char_blobs(bwA,h)
        bw = bwB; 
    else
        bw = bwA; 
    end
end

% Filters out excessive background noise by deleting blobs exceeding total area percentage
function bw = remove_large_blobs(bw)
    cc = bwconncomp(bw);
    for ci=1:cc.NumObjects
        if length(cc.PixelIdxList{ci}) > 0.40*numel(bw)
            bw(cc.PixelIdxList{ci}) = 0; 
        end
    end
end

% Counts valid isolated character regions to determine optimal binarization selection
function n = count_char_blobs(bw, plate_h)
    n=0; 
    if isempty(bw), return; end
    st = regionprops(bw,'BoundingBox');
    for j=1:length(st)
        cb=st(j).BoundingBox; 
        c_ar=cb(3)/cb(4); 
        chr=cb(4)/plate_h;
        if c_ar>0.07 && c_ar<2.5 && chr>0.13 && chr<0.95 && cb(3)>3 && cb(4)>4
            n=n+1; 
        end
    end
end

% Extracts character bounding geometries, discarding framing components and central dividers
function raw_chars = segment_characters(plate_bw)
    raw_chars = [];
    if isempty(plate_bw), return; end
    h_plate = size(plate_bw,1);
    w_plate = size(plate_bw,2);
    st = regionprops(plate_bw,'BoundingBox');
    
    MIN_WIDTH_RATIO  = 0.01;  
    MAX_WIDTH_RATIO  = 0.3;  
    MIN_HEIGHT_RATIO = 0.10;  
    MAX_HEIGHT_RATIO = 0.85;  
    ASPECT_RATIO_MIN = 0.05;  
    ASPECT_RATIO_MAX = 3.00;  

    boxes = [];
    for j=1:length(st)
        cb=st(j).BoundingBox; 
        c_ar=cb(3)/cb(4); 
        chr=cb(4)/h_plate;
        c_w = cb(3)/w_plate;
        
        center_x = cb(1) + cb(3)/2;
        center_y = cb(2) + cb(4)/2;
        
        if center_y < h_plate * 0.3
            continue; 
        end
        
        if center_x < w_plate * 0.03 || center_x > w_plate * 0.97
            continue; 
        end
        
        if c_w < 0.015 && center_x > w_plate*0.48 && center_x < w_plate*0.52
            continue; 
        end
        
        if c_w >= MIN_WIDTH_RATIO && c_w <= MAX_WIDTH_RATIO && ...
           chr >= MIN_HEIGHT_RATIO && chr <= MAX_HEIGHT_RATIO && ...
           c_ar >= ASPECT_RATIO_MIN && c_ar <= ASPECT_RATIO_MAX
       
            boxes = [boxes; cb]; 
        end
    end
    raw_chars = boxes;
end