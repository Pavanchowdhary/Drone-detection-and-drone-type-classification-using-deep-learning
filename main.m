clc;clear;close all hidden;

data = load('glabels.mat');  %Load variables from file into workspace
S = dir(fullfile('C:\Users\DILEEP ROYAL\Downloads\Project\TK59560\CODE\DATASET\*.jpg'));  %List folder contents
Extract_name = extractfield(S,'name');  %Field values from structure array
Extract_name = Extract_name';
Extract_folder = extractfield(S,'folder');
Extract_folder = Extract_folder';
merge_cell = fullfile(Extract_folder,Extract_name);  %Build full file name from parts
merge_cell_table = cell2table(merge_cell);  %Convert cell array to table


data_drone = data.gTruth.ROILabelData.DATASET.drone;
data_drone = cell2table(data_drone);
trainingData = [merge_cell_table,data_drone];
shuffledIdx = randperm(height(trainingData));
trainingData = trainingData(shuffledIdx,:);

imds = imageDatastore(trainingData.merge_cell); %Datastore for image data
blds = boxLabelDatastore(trainingData(:,2:end)); %Datastore for bounding box label data

ds = combine(imds, blds); %Convenience function for static .NET System.Delegate Combine method

load('Yolov2_lgraph.mat');
lgraph = Yolov2_lgraph.lgraph;
% analyzeNetwork(lgraph);

lgraph.Layers

options = trainingOptions('sgdm',...   %Options for training deep learning neural network
          'InitialLearnRate',0.001,...
          'Verbose',true,...
          'MiniBatchSize',16,...
          'MaxEpochs',30,...
          'Shuffle','never',...
          'VerboseFrequency',30,...
          'CheckpointPath',tempdir);

% [detector,info] = trainYOLOv2ObjectDetector(ds,lgraph,options);

load detector;
load info;
figure
plot(info.TrainingLoss)
grid on
xlabel('Number of Iterations')
ylabel('Training Loss for Each Iteration')

[file,path] = uigetfile('*.*','select a video');  %Open file selection dialog box
org_video = VideoReader([path,file]); %Create object to read video files
num_frames = org_video.NumberOfFrames; 

n=num_frames; 
for i = 1:n
    frames = read(org_video,i);
    figure(1);imshow(frames);title('Original Video');
end

for i = 1:n
    img = read(org_video,i);
    [bboxes,scores] = detect(detector,img);  %object detector 

        if(~isempty(bboxes))
            img = insertObjectAnnotation(img,'rectangle',bboxes,scores); %Annotate truecolor or grayscale image or video stream
        end
    figure(2);imshow(img);title('Drone Detection'); 
end
c = imcrop(img,bboxes);
figure;imshow(c);title('Detected Drone');
%%
% imwrite(c,'detected_drone_Quadcopter.jpg');
%%
c = imresize(c,[32 32]);
matlabroot = pwd;
digitDatasetPath = fullfile(matlabroot,'CLASSIFICATION_DATASET');
imds = imageDatastore(digitDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');

layers = [
    imageInputLayer([32 32 3])
    
    convolution2dLayer(3,32,'Stride',1,'Padding','same','Name','conv_1')
    reluLayer
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
    
    convolution2dLayer(3,64,'Stride',1,'Padding','same','Name','conv_2')
    reluLayer
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')
   
    convolution2dLayer(3,128,'Stride',1,'Padding','same','Name','conv_3')
    reluLayer
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_3')
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

options2 = trainingOptions('sgdm','Plots','training-progress','MaxEpochs',40,'initialLearnRate',0.001);
convnet = trainNetwork(imds,layers,options2);
YPred = classify(convnet,c);
output=char(YPred);
msgbox(output)
clear
quit

