/opt/conda/bin/python main_yx.py --input-pc ./data/36_0_pc1.ply \
--initial-mesh ./data/36_0_initmesh.obj \
--save-path ./checkpoints/tooth2 \
--pools 0.1 0.0 0.0 0.0 \
--iterations 1000

#test
# /opt/conda/bin/python test.py --input-pc ./data/36_0_pc1.ply \
# --initial-mesh ./data/36_0_initmesh.obj \
# --save-path ./checkpoints/tooth2 \
# --load-model 1