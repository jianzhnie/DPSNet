NGPUS=2
### baseline
#python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "/home/jianzhnie/auto-check-out/dpsnet/configs_nie/e2e_faster_rcnn_R_101_FPN_1x_rpc_syn_render_density_map_v1.yaml"
#python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file "/home/jianzhnie/auto-check-out/dpsnet/configs_nie/e2e_faster_rcnn_R_101_FPN_1x_rpc_syn_render_density_map_v1_test.yaml"

python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "/home/jianzhnie/auto-check-out/dpsnet/configs_nie/e2e_faster_rcnn_R_101_FPN_1x_rpc_syn_render_density_map_v2.yaml"
#python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file "/home/jianzhnie/auto-check-out/dpsnet/configs_nie/e2e_faster_rcnn_R_101_FPN_1x_rpc_syn_render_density_map_v2_test.yaml"

##  crose-fintune
#python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "/home/jianzhnie/auto-check-out/dpsnet/configs_nie/e2e_faster_rcnn_R_101_FPN_1x_rpc_syn_render_cross_finetune_v1.yaml"
#python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file "/home/jianzhnie/auto-check-out/dpsnet/configs_nie/e2e_faster_rcnn_R_101_FPN_1x_rpc_syn_render_cross_finetune_v1_test.yaml"

#python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "/home/jianzhnie/auto-check-out/dpsnet/configs_nie/e2e_faster_rcnn_R_101_FPN_1x_rpc_syn_render_cross_finetune_v3.yaml"
#python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file "/home/jianzhnie/auto-check-out/dpsnet/configs_nie/e2e_faster_rcnn_R_101_FPN_1x_rpc_syn_render_cross_finetune_v3_test.yaml"


#python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "/home/jianzhnie/auto-check-out/dpsnet/configs_nie/e2e_faster_rcnn_R_101_FPN_1x_rpc_syn_render_cross_finetune_v4.yaml"
#python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file "/home/jianzhnie/auto-check-out/dpsnet/configs_nie/e2e_faster_rcnn_R_101_FPN_1x_rpc_syn_render_cross_finetune_v3_test.yaml"


# final-fintune
#python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "/home/jianzhnie/auto-check-out/dpsnet/configs_nie/e2e_faster_rcnn_R_101_FPN_1x_rpc_final_finetune_1_v1.yaml"
#python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "/home/jianzhnie/auto-check-out/dpsnet/configs_nie/e2e_faster_rcnn_R_101_FPN_1x_rpc_final_finetune_1_v2.yaml"


## use real data train and test
#python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "/home/jianzhnie/auto-check-out/dpsnet/configs_nie/e2e_faster_rcnn_R_101_FPN_1x_rpc_syn_render_density_map_v2.yaml"
