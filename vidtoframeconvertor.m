% import the video file
obj = VideoReader('2nd vid Trim.mp4');
vid = read(obj);

% read the total number of frames
frames = obj.NumFrames;

% file format of the frames to be saved in
ST ='.jpg';

% reading and writing the frames
for x = 1 : frames

	% converting integer to string
	Sx = num2str(x+163);

	% concatenating 2 strings
	Strc = strcat(Sx, ST);
	Vid = vid(:, :, :, x);
	cd ('Z:\GURU DARSHAN\2021\MATLAB\GROWTH PROJECTS\DRONE DETECTION AND CLASSIFICATION\CODE\SecondVideo');

	% exporting the frames
	imwrite(Vid, Strc);
	cd ..
end
