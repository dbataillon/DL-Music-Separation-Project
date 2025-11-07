# Music-Separation
This is an implementation of U-Net for vocal, bass, drums separation with tensorflow

## Requirement
- librosa==0.6.2
- numpy==1.14.3
- tensorflow==1.13.0
- python==3.6.5

## Download Dataset
I download [dsd100](https://sigsep.github.io/datasets/dsd100.html) dataset.
<pre><code>$ python download_data.py --DATADIR ./data </code></pre>

## Data
I prepare CCMixter datasets in "./data" and Each track consisted of Mixed, bass, drums, other, vocal version
<pre><code>$ python CCMixter_process.py --DATADIR ./data </code></pre>

## Usage
- Train (run everything from repo root and import via packages)
<pre><code>$ python -m training.Training.py</code></pre>
- Test
<pre><code>$ python -m evaluation.Test.py</code></pre>

## Paper
Jaehoon Oh et al. [spectrogram-channels u-net: a source separation model viewing each channel as the spectrogram of each source](https://arxiv.org/abs/1810.11520)

## Base Implimentation
* https://github.com/Jeongseungwoo/Singing-Voice-Separation

## To Do List
* convert wav files to mp3 files
* make tfrecord format files
