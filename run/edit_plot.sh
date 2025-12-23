python two_axis_editor.py   --input ./Data/csv/251101_stepover.csv --output ./Data/out.csv \
  --joint right_hip --axes wx wz --frame-range 0 297 \
  --op scale --sx 0.65 --sy 0.5  --anchor start

python two_axis_editor.py   --input ./Data/out.csv --output ./Data/out.csv \
  --joint right_hip --axes wy wz --frame-range 0 297 \
  --op scale --sx 0.75 --sy 1  --anchor start

python two_axis_editor.py   --input ./Data/out.csv --output ./Data/out.csv \
  --joint right_knee --axes wx wz --frame-range 0 297 \
  --op scale --sx 0.5 --sy 0.65  --anchor mean

python two_axis_editor.py   --input ./Data/out.csv --output ./Data/out.csv \
  --joint right_hip --axes wy wz --frame-range 0 297 \
  --op scale --sx 0.75 --sy 1  --anchor start

python two_axis_editor.py  --input ./Data/out.csv --output ./Data/out.csv \
  --joint right_ankle --axes wx wy --frame-range 0 297 \
  --op scale --sx 0.55 --sy 0.6  --anchor mean

python two_axis_editor.py  --input ./Data/out.csv --output ./Data/out.csv \
  --joint right_ankle --axes wy wz --frame-range 0 297 \
  --op scale --sx 1 --sy 0.65  --anchor mean

python csv2npz.py

python plot_graph.py