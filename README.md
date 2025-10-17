# Applying CTC Alignment on Earnings Conference Calls

## Overview
This project provides a simple and step-by-step instruction to apply CTC alignment on transcripts and recordings of earnings conference calls.

## Data
You would need two files for a call to implement the alignment:
1. The complete transcript of a call (see an example [here](data/test_txt.txt))
2. The complete recording of a call (see an example [here](data/test_wav.wav))

Make sure that your recording file is in `.wav` format since the library relies on the `WAV2VEC` algo. Refer to [`format_convert`](format_convert.py) for a quick implementation of 
`.mp3` to `.wav` conversion.

Additionally, if you have your transcript diarised in a similar format like this

| node | speaker id | text |
|-----------|-----------|-----------|
| 1    | 0    | Hello and welcome! ...    |
| 2    | 1    | Thank you. So ...    |

Then this tutorial will also help you extract slices of a recording corresponding to each node of the call.

## Workflow
   **IMPORTANT**
   _Check the dependencies first before you move on!_

The current repo is largely based on [this tutorial](https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html). I highly recommend reading that first before replicating this example.
Each step in the following workflow is largely independent of the other, so you may adapt it to your own needs and data structure (For steps 2 and 3, refer to the [`test`](test.py) file).

1. **Alignment at word level**
   With the transcript and recording ready under the `data` folder under cwd, we perform CTC alignment with the help of the [`torchaudio`](https://github.com/pytorch/audio) library, which is open-sourced and user-friendly.
   Use the [`gen_segments`](run_alignment.py#L12) function in the [`run_alignment`](run_alignment.py) module to generate a list of [`Segment`](DataClass.py#L10) objects, which have the following attributes:
   - `label`(str): the word
   - `start`(int): the start frame
   - `end`(int): the end frame
   - `score`(float): the score of stay vs change

   The test section in this module shows how to run the simple execution and save the output to a JSON file for later use. The output based on the sample transcript and recording can be found [here](data/test_segments.json). Here is a snapshot:
 ```
[
    {
        "label": "GOOD",
        "start": 0,
        "end": 49,
        "score": 0.8936395182260027
    },
    {
        "label": "DAY",
        "start": 52,
        "end": 64,
        "score": 0.6950824446200082
    },
    {
        "label": "AND",
        "start": 73,
        "end": 79,
        "score": 0.4499109933773677
    },
    {
        "label": "WELCOME",
        "start": 80,
        "end": 113,
        "score": 0.6528035912408747
    }, ...
 ```

2. **Obtain timestaps**

   It may be of more use if we know the start and end seconds of a word instead of the frame representation. Use the [`gen_timstamp`](jsonUtils.py#L11) function to do this. Note that the output is a list of [`WordStamp`](DataClass.py#L24) objects, which are
   similar to the `Segment` data class, but do not have the `score` attribute. Now we got the starts and ends for words in the transcript in a sequential order.

3. **Extract slices**

   If the transcript has been diarised, then you may extract the recording slices of each speaker with the help of [`gen_speech_timestamp`](utils.py#L8). The output would be a pandas df like:

| node | speaker id | text | start_sec | end_sec |
|-----------|-----------|-----------|-----------|-----------|
| 1    | 0    | Hello and welcome! ...    | 0.0 | 11.4|
| 2    | 1    | Thank you. So ...    | 51.4 | 191.9|

The slices will be saved to a local folder if the `save_folder` arg is given. The output of the sample data can be found [here](data/test_aligned_content.csv). 
All the slices corresponding to the sample call are saved in the `data/test_align` folder. You may check the `.csv` file with the `.wav` files to check whether the slice correctly captures the content of one's speech.


## Dependencies and environment
The alignment here mainly leverages `torchaudio`, and as a required part for audio processing, please check whether [ffmpeg](https://www.ffmpeg.org/) is well-functioning in your terminal. Please note that `ffmpeg v8.0` seems incompatible with current versions of `torchaudio`. 
Pay attention to the version you are employing.

I implemented the alignment for this tutorial on an NVIDIA A100 GPU with 8 CPUs and 64 GB RAM in the preset `pytorch` environment (for the versions of the packages, check [here](https://scrp.econ.cuhk.edu.hk/info/pytorch-list)) through the [SCRP](https://scrp.econ.cuhk.edu.hk/) under the Department of Economics, The CU of HK. 
I accessed the service on 15/10/2025.

