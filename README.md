# ActivityNet and kinetics400
Downloaded from https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics. I've been following the tutorial https://pytorchvideo.org/docs/tutorial_classification, which uses the Kinetics dataset, however, I have not been able to download the dataset. 



# videos
I am currently following the tutorial https://github.com/voxel51/fiftyone-examples/blob/master/examples/pytorchvideo_tutorial.ipynb, and outputted the created videos to /videos.



# bobsl
Currently when I try to download the bobsl dataset using the below command, the output is the videos file in /bobsl. I'm not sure what to do with this file.

"wget --recursive --no-parent --continue --wait=1 --no-host-directorie --cut-dirs 2 --user bobsl-00019 --password toh7jaib https://thor.robots.ox.ac.uk/~vgg/data/bobsl/videos"



# videos original and WSASLInfo
/videos_original and WSASLInfo contains the WSASL dataset and info downloaded by following the WSASL download process outlined in https://github.com/gulvarol/bsl1k. By default they are separated into train, test and val.



# reallocateWSASLDataset
This python script separates the above WSASl data by each sign. To run this script, a directory called "other" needs to be created in videos_original that should contain all videos from train, test and val.