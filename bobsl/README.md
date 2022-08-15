# BOBSLv1_2 dataset

This README accompanies BOBSLv1_2, an updated version of BOBSL_v1_1. For a detailed description of the changes, see `CHANGELOG_v1_2.md` BOBSL_v1_2 was released in August 2022.

For further information about BOBSL and the technical report describing the dataset, see
https://www.robots.ox.ac.uk/~vgg/data/bobsl/

## Dataset structure

This folder contains the BOBSL dataset.  The structure is as follows:

```bash
README.md  # this readme file
validation_script.py  # simple utilities for inspecting the dataset
videos/
  # contains 2212 mp4 videos
pose/
  # contains 2212 .tar files (holding openpose)
flow/
  # contains 2212 .tar files (holding raft flow)
features/
  i3d_c2281_16f_m8_-15_4_d0.8_-3_22/
     # contains 2212 files containing features extracted from each episode at a stride of 4 frames using an i3d model training on BOBSL with mouthings (confidence 0.8) and dictionary spottings (confidence 0.8) spanning 2281 classes
subtitles/
  audio-aligned/
    # audio-aligned subtitles for 1,940 episodes
  audio-aligned-heuristic-correction/
    # audio-aligned subtitles for 1,940 episodes that incorporate heuristic corrections to improve the alignment
  manually-aligned/
    # manually aligned subtitles for 55 episodes
spottings/
  mouthing_spottings.json # mouthing keyword spottings
  dict_spottings.json  # dictionary keyword spottings
  attention_spottings.json  # attention-based sign spottings
  verified_mouthing_spottings.json  # verified mouthing keyword spottings
  verified_dict_spottings.json  # verified dictionary keyword spottings
metadata_public_episodes.tsv # metadata associated with the BOBSL episodes
subset2episode.json # a mapping from subset names to episodes
```

Note that the 272 challenge episodes that are new to BOBSLv1_2 (and which form the basis of the 2022 ECCV workshop challenge) are not accompanied by annotations or metadata.

### Printing statistics

Use the following commands to print statistics:

```bash
# report statistics (note that different annotation sources tend to have different confidences)
SPOTTING_DIR="./spottings" # path to your spottings folder
PROB_THR=0.0 # select a minimum probability threshold to report stats over
COUNT_THR=1 # select a minimum number of occurences for the keyword to be included in the stats
python validation_script.py --spotting_dir $SPOTTING_DIR --prob_thrs $PROB_THR --count_thrs $COUNT_THR

Expected output:
Reporting statistcs for spottings/mouthing_spottings.json
Found 698415 spottings with conf 0.0 with effective vocab 22168 [each has at least 1 spottings] in train
Found 15266 spottings with conf 0.0 with effective vocab 3972 [each has at least 1 spottings] in val
Found 97296 spottings with conf 0.0 with effective vocab 10354 [each has at least 1 spottings] in test
Reporting statistcs for spottings/dict_spottings.json
Found 4958355 spottings with conf 0.0 with effective vocab 6643 [each has at least 1 spottings] in train
Found 121668 spottings with conf 0.0 with effective vocab 3869 [each has at least 1 spottings] in val
Found 918232 spottings with conf 0.0 with effective vocab 5542 [each has at least 1 spottings] in test
Reporting statistcs for spottings/attention_spottings.json
Found 429484 spottings with conf 0.0 with effective vocab 1409 [each has at least 1 spottings] in train
Found 9346 spottings with conf 0.0 with effective vocab 649 [each has at least 1 spottings] in val
Found 51917 spottings with conf 0.0 with effective vocab 987 [each has at least 1 spottings] in test
```
