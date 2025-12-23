python 11aa_to_euler_csv.py \
  --input  Data/csv/sensing_SO_local_1204.csv \
  --output Data/csv/edited_s_SO_local_1204.csv \
  --aa-suffixes wx wy wz \
  --euler-suffixes ex ey ez \
  --drop-aa

### HIP ###
# flexion extension 조절
python 1axis.py \
  --input Data/csv/edited_s_SO_local_1204.csv \
  --output Data/csv/1axis_s_SO_local_1204.csv \
  --joint right_hip \
  --axis ez \
  --frame-range 0 60 \
  --op scale \
  --scale 0.6 \
  --anchor start \
  --smooth-window 5
# abduction adduction 조절1
python 1axis.py \
  --input Data/csv/1axis_s_SO_local_1204.csv \
  --output Data/csv/1axis_s_SO_local_1204.csv \
  --joint right_hip \
  --axis ey \
  --frame-range 0 60 \
  --op scale \
  --scale 1.3 \
  --anchor start \
  --smooth-window 5
# abduction adduction 조절2
python 1axis.py \
  --input Data/csv/1axis_s_SO_local_1204.csv \
  --output Data/csv/1axis_s_SO_local_1204.csv \
  --joint right_hip \
  --axis ey \
  --frame-range 0 30 \
  --op scale \
  --scale 0.6 \
  --anchor start \
  --smooth-window 5
# translate 적용
python 1axis.py \
  --input Data/csv/1axis_s_SO_local_1204.csv \
  --output Data/csv/1axis_s_SO_local_1204.csv \
  --joint right_hip \
  --axis ey \
  --frame-range 0 60 \
  --op translate \
  --dt -7
python 1axis.py \
  --input Data/csv/1axis_s_SO_local_1204.csv \
  --output Data/csv/1axis_s_SO_local_1204.csv \
  --joint right_hip \
  --axis ez \
  --frame-range 0 60 \
  --op translate \
  --dt 7

### KNEE ###
# translate 적용
python 1axis.py \
  --input Data/csv/1axis_s_SO_local_1204.csv \
  --output Data/csv/1axis_s_SO_local_1204.csv \
  --joint right_knee \
  --axis ez \
  --frame-range 0 60 \
  --op translate \
  --dt -7
# flextion extension 조절 (KF-4)
python 1axis.py \
  --input Data/csv/1axis_s_SO_local_1204.csv \
  --output Data/csv/1axis_s_SO_local_1204.csv \
  --joint right_knee \
  --axis ez \
  --frame-range 16 50 \
  --op scale \
  --scale 1.1 \
  --anchor start \
  --smooth-window 3
# flextion extension 조절 (KF-2)
python 1axis.py \
  --input Data/csv/1axis_s_SO_local_1204.csv \
  --output Data/csv/1axis_s_SO_local_1204.csv \
  --joint right_knee \
  --axis ez \
  --frame-range 0 60 \
  --op scale \
  --scale 0.8 \
  --anchor start \
  --smooth-window 5

### ANKLE ###
python 222KF_editor_euler.py \
  --input Data/csv/1axis_s_SO_local_1204.csv \
  --output Data/csv/1axis_s_SO_local_1204.csv \
  --cols right_ankle_ex right_ankle_ey right_ankle_ez \
  --mode pivot \
  --keyframe 7-16-50:,,15

python 1axis.py \
  --input Data/csv/1axis_s_SO_local_1204.csv \
  --output Data/csv/1axis_s_SO_local_1204.csv \
  --joint right_ankle \
  --axis ez \
  --frame-range 0 60 \
  --op scale \
  --scale 0.7 \
  --anchor start \
  --smooth-window 5

python 33euler_to_aa_csv.py \
  --input  Data/csv/1axis_s_SO_local_1204.csv \
  --output Data/csv/edited_1axis_s_SO_local_1204.csv \
  --euler-suffixes ex ey ez \
  --aa-suffixes wx wy wz

python csv2npz.py

python plot_graph.py

#  1  2  3   4   5   6   // end: 60
# [0, 8, 16, 29, 43, 50] 