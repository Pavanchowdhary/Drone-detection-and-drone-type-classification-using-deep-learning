clear all;close all;clc;
srcFiles = dir('U:\GURU DARSHAN\GURU DARSHAN\2021\MATLAB\GROWTH PROJECTS\TMMAIP382 CANCER CELL DETECTION\CNN\Dataaaaa\Abnormal\*.jpg');  % the folder in which ur images exists
for i = 1 : length(srcFiles)
filename = strcat('U:\GURU DARSHAN\GURU DARSHAN\2021\MATLAB\GROWTH PROJECTS\TMMAIP382 CANCER CELL DETECTION\CNN\Dataaaaa\Abnormal\',srcFiles(i).name);
im = imread(filename);
im = imresize(im,[32 32]);
newfilename=fullfile('U:\GURU DARSHAN\GURU DARSHAN\2021\MATLAB\GROWTH PROJECTS\TMMAIP382 CANCER CELL DETECTION\CNN\Dataset\Abnormal\',srcFiles(i).name);
imwrite(im,newfilename,'jpg');
end

