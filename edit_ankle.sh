# python 11aa_to_euler_csv.py \
#   --input  Data/csv/1axis_s_SO_local_1204.csv \
#   --output Data/csv/1axis_s_SO_local_1204.csv \
#   --aa-suffixes wx wy wz \
#   --euler-suffixes ex ey ez \
#   --drop-aa

# X Y Z
# Y Z X

# python 222KF_editor_euler.py \
#   --input Data/csv/edited_s_SO_local_1204.csv \
#   --output Data/csv/e_edited_s_SO_local_1204.csv \
#   --cols right_hip_ex right_hip_ey right_hip_ez \
#   --mode pivot \
#   --keyframe 16-29-43:,,-45

python 222KF_editor_euler.py \
  --input Data/csv/1axis_s_SO_local_1204.csv \
  --output Data/csv/1axis_s_SO_local_1204.csv \
  --cols right_ankle_ex right_ankle_ey right_ankle_ez \
  --mode pivot \
  --keyframe 7-16-43:,,15

python 33euler_to_aa_csv.py \
  --input  Data/csv/1axis_s_SO_local_1204.csv \
  --output Data/csv/edited_1axis_s_SO_local_1204.csv \
  --euler-suffixes ex ey ez \
  --aa-suffixes wx wy wz

python csv2npz.py
python plot_graph.py

#  1  2  3   4   5   6   // end: 60
# [0, 8, 16, 29, 43, 50] 

# python 22keyframe_editor_euler.py \
#   --input Data/csv/edited_s_SO_local_1204.csv \
#   --output Data/csv/e_edited_s_SO_local_1204.csv \
#   --cols right_hip_ex right_hip_ey right_hip_ez \
#   --mode pivot \
#   --keyframe 0-29-59:,,-45
#   # --keyframe 16-29-43:,,-45
# --mode local_bump \

# --keyframe 30:45 \
# --radius 10
# python 222KF_editor_euler.py \
#   --input Data/csv/edited_s_SO_local_1204.csv \
#   --output Data/csv/e_edited_s_SO_local_1204_local.csv \
#   --cols right_hip_ex right_hip_ey right_hip_ez \
#   --mode local \
#   --local-radius 10 \
#   --keyframe 30:45,,