_base_ = ['/root/yt/paper/mmrazor/configs/sub_set/step2_sub_test.py']

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
)

algorithm = dict(bn_training_mode=True)

## SEARCHER
searcher = dict(
    type='EvolutionSearcher',
    metrics='bbox',
    score_key='bbox_mAP',
    constraints=dict(flops=30000 * 1e6),
    candidate_pool_size=5,
    candidate_top_k=5,
    max_epoch=3,
    num_mutation=5,
    num_crossover=5
)





