%% STEMorph Emotional Rating Task
%
% This script runs an experiment where participants rate the emotional
% valence of facial expressions on a 9-point Likert scale.
%
% Inputs:
%   - Facial expression images in the 'Faces' folder
%   - User input for participant ID
%
% Outputs:
%   - CSV file with participant responses and trial information
%   - MAT file with all experiment data
%
% Dependencies:
%   - Psychtoolbox
%
% Authors: Mohammad Ebrahim Katebi, Mohammad Hossein Ghafari, Tara Ghafari

%% Clear the workspace and the screen
sca;
close all;
clear;

Screen('Preference', 'SkipSyncTests', 1);

% Experiment parameters
angel_num = 2; % Here: repeat_num
morph_step_num = 9; % Number of morphing steps
repeat_num = 22; % Here: Number of face identities

run_num = repeat_num * angel_num * morph_step_num;

small_break_interval = run_num / 3; % 2 Min
big_break_interval = run_num + 1; % 10 Min

% Screen setup
PsychDefaultSetup(2);
cfgScreen.scrNum = max(Screen('Screens'));
[cfgScreen.dispSize.width, cfgScreen.dispSize.height] = Screen('DisplaySize', cfgScreen.scrNum);
cfgScreen.distance = 50;  % Distance from participant to the monitor in cm
cfgScreen.resolution = Screen('Resolution', cfgScreen.scrNum);
cfgScreen.fullScrn = [0, 0, cfgScreen.resolution.width, cfgScreen.resolution.height];

white = WhiteIndex(cfgScreen.scrNum);
black = BlackIndex(cfgScreen.scrNum);
grey = (white - black) / 2;
cfgScreen.backgroundColor = grey;

% Convert visual angles to pixels
peri_pix = angle2pix(cfgScreen, 0);
Face_height = angle2pix(cfgScreen, 9.11);

% Timing parameters
response_timeout = 4;

% Keyboard setup
KbName('UnifyKeyNames');
Keyboard.quitKey = KbName('ESCAPE');
Keyboard.confirmKey = KbName('c');
Keyboard.Key1 = KbName('1!'); % Angry
Keyboard.Key2 = KbName('2@');
Keyboard.Key3 = KbName('3#');
Keyboard.Key4 = KbName('4$');
Keyboard.Key5 = KbName('5%');
Keyboard.Key6 = KbName('6^');
Keyboard.Key7 = KbName('7&');
Keyboard.Key8 = KbName('8*');
Keyboard.Key9 = KbName('9('); % Happy

% Image processing parameters
Noise_Image_blockSize = 4;

% Participant information input
prompt = {'Enter ID:'};
defaults = {''};
opts.Interpreter = 'tex';
dims = [1 40];
answer = inputdlg(prompt, 'Info', dims, defaults, opts);
subject = answer{1,:};
clock_info = clock;
output_name = [subject '_' num2str(clock_info(2)) '_' num2str(clock_info(3)) '_' num2str(clock_info(4)) '_' num2str(clock_info(5))];

% Calculate face positions
angels = zeros(angel_num, 1);
face_pos = zeros(angel_num, 2);

for i = 0:(angel_num-1)
    interval = 360 / angel_num;
    angels(i+1) = i * interval;
    face_pos(i+1,1) = int32(peri_pix * cosd(angels(i+1)));
    face_pos(i+1,2) = int32(peri_pix * sind(angels(i+1)));
end

% Load face images
faces = dir('Faces/*.png');
face_names = {faces.name};
faces_num = size(face_names, 2);
face_names_cell = cellstr(face_names);

% Generate randomized trial sequence
run_seq = zeros(run_num, 9); % State, Position ID, Morph Step, Face Person, ITI, Trial_Onset, Stim_Onset, Stim_Offset, RT
morph_step_random = [];

for j = 1:run_num
    run_seq(j,2) =  ceil(j / (morph_step_num * repeat_num));

    if (ceil(j / (morph_step_num * repeat_num)) ~= ceil((j-1) / (morph_step_num * repeat_num)))
        face = zeros((repeat_num * morph_step_num), 2); % Morph Step, Face Person

        for k = 1:(repeat_num * morph_step_num)
            face(k,1) =  ceil(k / repeat_num);

            if (ceil(k / repeat_num) ~= ceil((k-1) / repeat_num))
                for l = 0:(repeat_num-1)
                    face(k+l,2) =  (l+1);
                end
            end
        end

        ir_f = randperm(size(face,1));
        face_random = face(ir_f,:);
    end

    r = rem(j, (morph_step_num * repeat_num));
    if (r == 0)
        r = (morph_step_num * repeat_num);
    end

    run_seq(j,3) = face_random(r,1);
    run_seq(j,4) = face_random(r,2);
    run_seq(j,5) = rand + 1; % Jitters prestim interval between 1 and 2 seconds
end

% Randomize trial order
ir = randperm(size(run_seq,1));
run_seq_random = run_seq(ir,:);

% Add trial IDs
IDs = (1:run_num)';
run_seq_random = [IDs, run_seq_random];

% Load face images and create noise images
run_seq_random_face = cell(run_num,1);
run_seq_random_Noise = cell(run_num,1);

for p = 1:run_num
    face_name = strcat('_', num2str(run_seq_random(p,5),'%02.0f'), '_', num2str(run_seq_random(p,4),'%01.0f'), '.png');
    face_search_indices = ~cellfun(@isempty, regexp(face_names_cell, strcat('[MF]', face_name)));

    if(sum(face_search_indices)~=1)
        run_seq_random(p,2) = 2; % 2: Face not found
    elseif (sum(face_search_indices)==1)
        face_name = face_names_cell{face_search_indices==1};
        face_image_location = strcat('Faces/', face_name);
        [run_seq_random_face{p,1}, ~, alpha] = imread(face_image_location);
        run_seq_random_face{p,1}(:, :, 4) = alpha;

        % Create noise image from previous face
        if (p>1)
            Noise_Image = run_seq_random_face{p-1,1};
            [s1, s2, s3] = size(Noise_Image);
            nRows = round(s1 / Noise_Image_blockSize);
            nCols = round(s2 / Noise_Image_blockSize);
            Noise_Image = mat2cell(Noise_Image, ones(1, nRows) * Noise_Image_blockSize, ones(1, nCols) * Noise_Image_blockSize, s3);
            Noise_Image = cell2mat(reshape(Noise_Image(randperm(nRows * nCols)), nRows, nCols));
            run_seq_random_Noise{p,1} = Noise_Image;
        end
    end
end

run_seq_random_key = strings([run_num,1]); % Result Key

%% Set up Psychtoolbox screen
PsychDefaultSetup(2);
screens = Screen('Screens');
screenNumber = max(screens);
[window, windowRect] = PsychImaging('OpenWindow', screenNumber, grey);

% Set maximum priority for the window
topPriorityLevel = MaxPriority(window);
Priority(topPriorityLevel);

% Get screen properties
[screenXpixels, screenYpixels] = Screen('WindowSize', window);
ifi = Screen('GetFlipInterval', window);

% Set up alpha-blending for transparency
Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

% Set text properties
Screen('TextFont', window, 'Ariel');
Screen('TextSize', window, 32);

% Get the centre coordinate of the window
[xCenter, yCenter] = RectCenter(windowRect);

% Set up fixation cross
FixCross_DimPix = 16;
FixCross_xCoords = [-FixCross_DimPix FixCross_DimPix 0 0];
FixCross_yCoords = [0 0 -FixCross_DimPix FixCross_DimPix];
FixCross_allCoords = [FixCross_xCoords; FixCross_yCoords];
FixCross_lineWidthPix = 3;

% Set up fixation circle
FixCircle_lineWidthPix = 2;
FixCircle_baseRect = [-FixCross_DimPix -FixCross_DimPix FixCross_DimPix FixCross_DimPix];
FixCircle_centeredRect = CenterRectOnPoint(FixCircle_baseRect, xCenter, yCenter);

% Display start screen
DrawFormattedText(window, 'Press Anykey To Start :)', 'center', 'center', [1 1 1]);
Screen('Flip', window);
KbStrokeWait;

% Hide cursor and suppress keyboard input
ListenChar(-1);
HideCursor();

% Set up keyboard queue
Keyboard.activeKeys = [Keyboard.quitKey, Keyboard.confirmKey, Keyboard.Key1, Keyboard.Key2, Keyboard.Key3, Keyboard.Key4, Keyboard.Key5, Keyboard.Key6, Keyboard.Key7, Keyboard.Key8, Keyboard.Key9];
Keyboard.responseKeys = [Keyboard.Key1, Keyboard.Key2, Keyboard.Key3, Keyboard.Key4, Keyboard.Key5, Keyboard.Key6, Keyboard.Key7, Keyboard.Key8, Keyboard.Key9];
Keyboard.deviceNum = -1;

scanList = zeros(1,256);
scanList(Keyboard.activeKeys) = 1;
KbQueueCreate(Keyboard.deviceNum, scanList);
KbQueueStart;
KbQueueFlush;

%% Main experimental loop
Abortion = 0;
Abortion_Pauses = zeros(run_num,1);
Task_Onset = GetSecs();

for n = 1:run_num
    if (Abortion == 1)
        break;
    end

    if (run_seq_random(n,2) == 2)
        run_seq_random(n,10) = NaN;
        run_seq_random_key(n,1) = 'None';
        break;
    end

    % Check for breaks
    if ((ceil(n / big_break_interval) ~= ceil((n-1) / big_break_interval)) && n ~= 1)
        % 10-minute break
        DrawFormattedText(window, 'Break For 10 Min :)', 'center', 'center', [1 1 1]);
        vbl = Screen('Flip', window);
        DrawFormattedText(window, 'Press Anykey To Start :)', 'center', 'center', [1 1 1]);
        Screen('Flip', window, vbl + 600);
        KbStrokeWait;
    elseif ((ceil(n / small_break_interval) ~= ceil((n-1) / small_break_interval)) && n ~= 1)
        % 2-minute break
        DrawFormattedText(window, 'Break For 2 Min :)', 'center', 'center', [1 1 1]);
        vbl = Screen('Flip', window);
        DrawFormattedText(window, 'Press Anykey To Start :)', 'center', 'center', [1 1 1]);
        Screen('Flip', window, vbl + 120);
        KbStrokeWait;
    end

    % Start of trial
    Trial_Onset = GetSecs();
    Face_position_x = xCenter;
    Face_position_y = yCenter;

    % Display noise image
    if (n > 1)
        Noise_Image = run_seq_random_Noise{n,1};
        [s1, s2, ~] = size(Noise_Image);
        aspect_ratio = s2 / s1;
        Face_width = Face_height * aspect_ratio;
        Face_rec = [0 0 Face_width Face_height];
        Position_rec = CenterRectOnPoint(Face_rec, Face_position_x, Face_position_y);
        Noise_Face_texture = Screen('MakeTexture', window, Noise_Image);
        Screen('DrawTexture', window, Noise_Face_texture, [], Position_rec);
    end
    Screen('Flip', window);

    % Display face image
    Face_texture = Screen('MakeTexture', window, run_seq_random_face{n,1});
    [s1, s2, ~] = size(run_seq_random_face{n,1});
    aspect_ratio = s2 / s1;
    Face_width = Face_height * aspect_ratio;
    Face_rec = [0 0 Face_width Face_height];
    Position_rec = CenterRectOnPoint(Face_rec, Face_position_x, Face_position_y);
    Screen('DrawTexture', window, Face_texture, [], Position_rec);

    WaitSecs(run_seq_random(n,6)); % Jittered pre-stimulus interval
    Stim_Onset = Screen('Flip', window);

    % Wait for response
    KbQueueFlush;
    noResp = 1;
    while (noResp == 1)
        [presd, firstPrsd] = KbQueueCheck;
        keyCod = find(firstPrsd, 1);

        if (presd && (ismember(keyCod, Keyboard.responseKeys)))
            % Valid response
            Response_Key_time = firstPrsd(keyCod);
            thekeys = KbName(keyCod);
            thekeys = string(thekeys);

            run_seq_random(n,10) = Response_Key_time - Stim_Onset;
            run_seq_random_key(n,1) = thekeys;
            run_seq_random(n,2) = 1; % 1: Done
            noResp = 0;
            break;
        elseif (presd && keyCod == Keyboard.quitKey)
            % Abort experiment
            warning('Experiment Aborted!')
            Abortion_Pauses(n,1) = Abortion_Pauses(n,1) + 1;

            DrawFormattedText(window, 'Press C To Confirm :)', 'center', 'center', [1 1 1]);
            Screen('Flip', window);

            [~, abrtPrsd] = KbStrokeWait;
            if abrtPrsd(Keyboard.confirmKey)
                Abortion = 1;
                run_seq_random(n,2) = 4; % 4: Abortion
                run_seq_random(n,10) = NaN;
                run_seq_random_key(n,1) = 'None';
                noResp = 0;
                break;
            end

            % Resume experiment if not confirmed
            Screen('Flip', window, 0, 1);
            Screen('DrawTexture', window, Face_texture, [], Position_rec);
            WaitSecs(run_seq_random(n,6));
            Stim_Onset = Screen('Flip', window);
            KbQueueFlush;
        elseif ((GetSecs - Stim_Onset) > response_timeout)
            % No response within timeout
            run_seq_random(n,2) = 3; % 3: No Answer
            run_seq_random(n,10) = NaN;
            run_seq_random_key(n,1) = 'None';
            noResp = 0;
            break;
        end
    end

    % Record trial timings
    run_seq_random(n,7) = Trial_Onset;
    run_seq_random(n,8) = Stim_Onset;
end

% Display exit screen
DrawFormattedText(window, 'Press Anykey To Exit :)', 'center', 'center', [1 1 1]);
Screen('Flip', window);
KbStrokeWait;

%% Clean up
ListenChar(0);
ShowCursor();
Screen('CloseAll');
sca;

% Process and save results
run_seq_random_sorted = sortrows(run_seq_random, [3 4 5]);
run_seq_random_table = array2table(run_seq_random, "VariableNames", ["ID", "State", "Position ID", "Morph Step", "Face Person", "ITI", "Trial_Onset", "Stim_Onset", "Stim_Offset", "RT"]);

% Clean up response keys
for q = 1:n
    if (run_seq_random(q,2) == 1)
        run_seq_random_key(q) = replace(run_seq_random_key(q), {'1!', '2@', '3#', '4$', '5%', '6^', '7&', '8*', '9('}, {'1', '2', '3', '4', '5', '6', '7', '8', '9'});
    end
end

run_seq_random_key_table = array2table(run_seq_random_key, "VariableNames", "Answer");
Output_table = [run_seq_random_table run_seq_random_key_table];

% Save results
writetable(Output_table, strcat('./Data/Subject_', output_name, '.csv'));
save(strcat('./Data/Subject_', output_name));

% Helper function to convert visual angles to pixels
function pixel = angle2pix(cfgScreen, angle)
pixSize = (cfgScreen.dispSize.width./10)/cfgScreen.resolution.width;
sz = 2 * cfgScreen.distance * tan(pi * angle / (2 * 180));
pixel = round(sz/pixSize);
end