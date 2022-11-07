%% Classify Videos using a pretrained GoogLeNet CNN combined with an LSTM network

%% Load Pretrained Convolutional Network
netCNN = googlenet;
%% Load Data

dataFolder = "D:\Uni\capstone\ASLLRP\train";
testFolder = "D:\Uni\capstone\ASLLRP\test";
[files,labels] = hmdb51Files(dataFolder);
%% 
% Read the first video and view the size and corresponding label of the video

idx = 1;
filename = files(idx);
video = readVideo(filename);
size(video)
labels(idx)
%% 
% View the video

numFrames = size(video,4);
figure
for i = 1:numFrames
    frame = video(:,:,:,i);
    imshow(frame/255);
    drawnow
end
%% Convert Frames to Feature Vectors
% Convert the videos to sequences using the CN as a feature extractor.
% The data is first resized to match the input size of the network

inputSize = netCNN.Layers(1).InputSize(1:2);
layerName = "pool5-7x7_s1";

tempFile = fullfile(tempdir,"hmdb51_org.mat");

if exist(tempFile,'file')
    load(tempFile,"sequences")
else
    numFiles = numel(files);
    sequences = cell(numFiles,1);
    
    for i = 1:numFiles
        fprintf("Reading file %d of %d...\n", i, numFiles)
        
        video = readVideo(files(i));
        video = centerCrop(video,inputSize);
        
        sequences{i,1} = activations(netCNN,video,layerName,'OutputAs','columns');
    end
    
    save(tempFile,"sequences","-v7.3");
end

%% Prepare Training Data
% Partition data into training and validation data

numObservations = numel(sequences);
idx = randperm(numObservations);
N = floor(0.85 * numObservations);

idxTrain = idx(1:N);
sequencesTrain = sequences(idxTrain);
labelsTrain = labels(idxTrain);

idxValidation = idx(N+1:end);
sequencesValidation = sequences(idxValidation);
labelsValidation = labels(idxValidation);

%% Create LSTM Network
% Network layers:
% Sequence input layer matching feature dimension of feature vectors
% BiLSTM layer with 2000 hidden units and dropout layer
% Fully connected layer with an output size corresponding to the number of classes
% Softmax layer
% Classification layer.

numFeatures = size(sequencesTrain{1},1);
numClasses = numel(categories(labelsTrain));

layers = [
    sequenceInputLayer(numFeatures,'Name','sequence')
    bilstmLayer(2000,'OutputMode','last','Name','bilstm')
    dropoutLayer(0.5,'Name','drop')
    fullyConnectedLayer(numClasses,'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classification')];
%% Specify Training Options

miniBatchSize = 16;
numObservations = numel(sequencesTrain);
numIterationsPerEpoch = floor(numObservations / miniBatchSize);

options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',1e-4, ...
    'GradientThreshold',2, ...
    'MaxEpochs',5, ... 
    'Shuffle','every-epoch', ...
    'ValidationData',{sequencesValidation,labelsValidation}, ...
    'ValidationFrequency',numIterationsPerEpoch, ...
    'Plots','training-progress', ...
    'Verbose',false);
%% Train LSTM Network

[netLSTM,info] = trainNetwork(sequencesTrain,labelsTrain,layers,options);
%% Calculate the classification accuracy of the network on the validation set. 

YPred = classify(netLSTM,sequencesValidation,'MiniBatchSize',miniBatchSize);
YValidation = labelsValidation;
accuracy = mean(YPred == YValidation)
%% Assemble Video Classification Network
% Transform the videos into vector sequences using CN, then classify using
% layers from LSTM network.

% Add Convolutional Layers
cnnLayers = layerGraph(netCNN);

% Remove the input layer and the layers after the pooling layer used 
% for the activations
layerNames = ["data" "pool5-drop_7x7_s1" "loss3-classifier" "prob" "output"];
cnnLayers = removeLayers(cnnLayers,layerNames);

% Add Sequence Input Layer
inputSize = netCNN.Layers(1).InputSize(1:2);
averageImage = netCNN.Layers(1).Mean;

inputLayer = sequenceInputLayer([inputSize 3], ...
    'Normalization','zerocenter', ...
    'Mean',averageImage, ...
    'Name','input');

% Add the sequence input layer to the layer graph.
layers = [
    inputLayer
    sequenceFoldingLayer('Name','fold')];

lgraph = addLayers(cnnLayers,layers);
lgraph = connectLayers(lgraph,"fold/out","conv1-7x7_s2");

% Add LSTM Layers
lstmLayers = netLSTM.Layers;
lstmLayers(1) = [];

layers = [
    sequenceUnfoldingLayer('Name','unfold')
    flattenLayer('Name','flatten')
    lstmLayers];

lgraph = addLayers(lgraph,layers);
lgraph = connectLayers(lgraph,"pool5-7x7_s1","unfold/in");
lgraph = connectLayers(lgraph,"fold/miniBatchSize","unfold/miniBatchSize");

analyzeNetwork(lgraph)
%% Assemble the network
net = assembleNetwork(lgraph)
%% Classify Using New Data

predictions = strings([100,3]);
testClasses = ["car" "deaf" "finish" "friend" "future" "go-out" "must" "now" "who" "why"];
for class = 1:length(testClasses)
    testFolder = testFolder + "\" + testClasses(class);
    testFiles = dir(testFolder + "\*.mp4");
    for fileIndex = 1:length(testFiles)
        prediction = strings([1,3]);
        fileName = testFiles(fileIndex).folder + "\" + testFiles(fileIndex).name
        video = readVideo(fileName);
        video = centerCrop(video,inputSize);
        x = {video};
        YPred = classify(net,{video});
        trueClass = split(testFiles(fileIndex).folder,'\');
        trueClass = categorical(trueClass(end));
    
        prediction(1,1) = fileName;
        prediction(1,2) = trueClass;
        prediction(1,3) = YPred;

        row = 10*(class-1)+fileIndex;
        predictions(row,:) = prediction;
    end
end
%% 
%% Helper Functions

% read in video file and return |H|-by-|W|-by-|C-|by-|S| array, where
% |H|, |W|, |C|, and |S| are the height, width, number of channels, 
% and number of frames of the video, respectively.
function video = readVideo(filename)

vr = VideoReader(filename);
H = vr.Height;
W = vr.Width;
C = 3;

% Preallocate video array
numFrames = floor(vr.Duration * vr.FrameRate);
video = zeros(H,W,C,numFrames);

% Read frames
i = 0;
while hasFrame(vr)
    i = i + 1;
    video(:,:,:,i) = readFrame(vr);
end

% Remove unallocated frames
if size(video,4) > i
    video(:,:,:,i+1:end) = [];
end

end
%% 
% Crop the longest edges of a video and resize to input size
function videoResized = centerCrop(video,inputSize)

sz = size(video);

if sz(1) < sz(2)
    % Video is landscape
    idx = floor((sz(2) - sz(1))/2);
    video(:,1:(idx-1),:,:) = [];
    video(:,(sz(1)+1):end,:,:) = [];
    
elseif sz(2) < sz(1)
    % Video is portrait
    idx = floor((sz(1) - sz(2))/2);
    video(1:(idx-1),:,:,:) = [];
    video((sz(2)+1):end,:,:,:) = [];
end

videoResized = imresize(video,inputSize(1:2));

end

%%
% Return a list of files and labels from dataset in dataFolder
function [files, labels] = hmdb51Files(dataFolder)

fileExtension = ".mp4";
listing = dir(fullfile(dataFolder, "*", "*" + fileExtension));
numObservations = numel(listing);
numSubsetObservations = floor(numObservations);

if nargin == 2
    idx = randperm(numObservations,numSubsetObservations);
    listing = listing(idx);
end

files = strings(numSubsetObservations,1);
labels = cell(numSubsetObservations,1);

for i = 1:numSubsetObservations
    name = listing(i).name;
    folder = listing(i).folder;
    
    [~,labels{i}] = fileparts(folder);
    files(i) = fullfile(folder,name);
end

labels = categorical(labels);

end
