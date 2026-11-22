# Ego4D and Ego-Exo4D frames

Ego4D and Ego-Exo4D frames are not in the download; you get the videos from [ego4d-data.org](https://ego4d-data.org).
Every other source ships as images. For each Ego frame the dataset has a row in `ego/pointers.jsonl`:
the video in the official layout, the presentation timestamp of the keyframe, how it was resized or undistorted,
the edited pixels of each look as an RGBA patch with its box, and a 64-bit perceptual hash. `props rebuild-ego`
decodes the frames from your copy, pastes the patches, cuts the crops and checks every result against its hash.

## Steps

1. Request access at [ego4d-data.org](https://ego4d-data.org) and install their CLI (`pip install ego4d`).
2. Download the parts that PROPS uses:

   ```bash
   ego4d  --output_directory ~/ego4d    --datasets clips --version v2   # -> ~/ego4d/v2/clips/<uid>.mp4
   egoexo -o ~/egoexo4d --parts takes --views ego                     # -> ~/egoexo4d/takes/<take>/frame_aligned_videos/
   ```

3. Rebuild:

   ```bash
   pip install -e ".[ego]"        # PyAV and OpenCV (setup.sh --ego does this)
   props rebuild-ego --data data/PROPS-Benchmark --ego4d ~/ego4d --egoexo4d ~/egoexo4d --workers 16
   props rebuild-ego --data data/PROPS-Train     --ego4d ~/ego4d --egoexo4d ~/egoexo4d --ffmpeg $(which ffmpeg)
   ```

Only the videos that PROPS uses are opened: 532 for the benchmark and 767 for the training set. The report is
written to `ego/rebuild_report.json`, and the exit code is 1 if any image failed. Decoders differ a little between
versions, so a rebuilt JPEG is close to the released one but not identical; the check accepts a pHash Hamming
distance of up to 8 of 64 bits (`--max-dist`).

## What each pair gets

The `rebuild` field of `pairs.jsonl` says it:

| `rebuild` | pairs | meaning |
|---|---:|---|
| `images` | 1,123 | no Ego frame; usable right after the download (the `open` split) |
| `full` | 905 | every image rebuilds from the pointers |
| `no_gt` | 203 | every model input and reference crop rebuilds; the true target frames are relit Ego frames and are withheld. Scoring does not use them |
| `partial` | 26 | Resist-Intercut pairs whose history holds a relit Ego frame, which no pointer can rebuild |

The `full` and `phase_a` splits contain every pair except the 26 `partial` ones, so after the rebuild every pair
of these splits can be drawn and scored. When an input image is missing, `props export` lists how many frames it
blocks and `props generate` stops before drawing anything.
