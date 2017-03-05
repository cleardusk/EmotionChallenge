### For final evaluation

#### Submission
Just change to cnn dir and
```
python extract.py
```
It will load data preprocessed and caffe model to generate labels named `predictions.txt` and `predictions.zip` for test data. All details were considered.
 
Just upload `predictions.zip` to submit window then.

### Introduction
We use [Dlib](https://github.com/davisking/dlib) to do face and landmark detection, and use landmark to do face cropping and alignment, then we use [Caffe](https://github.com/BVLC/caffe) to with landmark and cropping image to train a cnn model to do the face expression recognition task.

### Pipline
#### Preproces
First, run `landmark.py` to get all the origin image landmark, then build the `crop_align` binary, and run `crop_align.py` to get all the 224x224 size image.

**Build crop_align**
 ```
 cd crop_align
 mkdir build
 cd build
 cmake ..
 make
 ```

All the preprocessed data except the images are in data dir.

#### Training
Change to cnn dir, run `prepare_data.py` to prepare training, validation and test data. Then run `train_val.sh` to start training.

#### Extract(Test)
Just run `extract.py` to generate the result, the input is the test image and its landmark offset info.

### Method
We use the landmark offset and image info to do this task. In detail, the landmark offset is calculated by substraction of 224x224 image landmark and each id's mean landmark, and we concact this feature to modified alexnet's last output feature. We change softmax loss to hinge loss to get a little better result.
  
  More detail is in the `fact_sheet.tex`.