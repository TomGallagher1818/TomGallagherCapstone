# ActivityNet and kinetics400
Downloaded ActivityNet repository from https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics. I've been following the tutorial https://pytorchvideo.org/docs/tutorial_classification, which uses the Kinetics dataset, however, I have not been able to download the dataset. 



# videos
I am currently following the tutorial https://github.com/voxel51/fiftyone-examples/blob/master/examples/pytorchvideo_tutorial.ipynb, and outputted the created videos to /videos.



# bobsl
Currently when I try to download the bobsl dataset using the below command, the output is the videos file in /bobsl. I'm not sure what to do with this file.

"wget --recursive --no-parent --continue --wait=1 --no-host-directorie --cut-dirs 2 --user bobsl-00019 --password toh7jaib https://thor.robots.ox.ac.uk/~vgg/data/bobsl/videos"




# reallocateWSASLDataset
This python script separates the above WSASL data by each sign. To run this script, first download the WSASL data by following the 'Preparing the data' section at https://github.com/gulvarol/bsl1k. After running the script 'download_wlasl.py', a videos_original directory is created that separates the WSASL videos into train, test and val. To run this script, a directory called "other" needs to be created in videos_original that should contain all videos from train, test and val.

# downloadSignBankVideos
This python script downloads all Auslan signed videos from https://auslan.org.au/dictionary/search/?query=A&category=all and adds them to SignBankVideos directory.


# testBsl1k data on WSASL
To run this script, you first need to download and setup the bslk data:
# Clone this repository
git clone https://github.com/gulvarol/bsl1k.git
cd bsl1k/
# Create bsl1k_env environment with dependencies
conda env create -f environment.yml
conda activate bsl1k_env
pip install -r requirements.txt

Within the bs1lk_env, running demo\demo.py should result in no issues.