#!/bin/bash
# Hard-cut segment generator for long H3 clips on a 16 GiB Mac.
# Each segment is an independent full pipeline run (peak = single segment), then
# concatenated with ffmpeg. Segments are NOT visually continuous (no vision
# encoder available for --first-frame), so use an incrementing seed to vary them.
set -e

MODEL=/Volumes/data/work/h3_qad/h3_int6g128_native
PROMPT="a calm ocean at sunset, gentle waves"
W=864
H=480
SEG_SECONDS=2          # ~56 aligned frames (~2.3 s) per segment
SEG_COUNT=7            # 7 x ~2.3 s ~= 16 s
OUTDIR=/Volumes/data/git/c/h3c/seg_out

mkdir -p "$OUTDIR"
cd /Volumes/data/git/c/h3c

for i in $(seq 1 "$SEG_COUNT"); do
    echo "=== segment $i/$SEG_COUNT ==="
    ./h3 -d "$MODEL" -p "$PROMPT" \
        --width "$W" --height "$H" \
        --seconds "$SEG_SECONDS" --steps 4 --token-reduction \
        --seed $((42 + i * 7)) \
        -o "$OUTDIR/seg_$i.mp4"
    echo "--- segment $i done ---"
done

cd "$OUTDIR"
: > list.txt
for i in $(seq 1 "$SEG_COUNT"); do echo "file 'seg_$i.mp4'" >> list.txt; done
ffmpeg -y -f concat -safe 0 -i list.txt -c copy final.mp4
echo "ALL DONE: $OUTDIR/final.mp4"
