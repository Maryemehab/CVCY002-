# Earlier run - SegFormer-B2, 4 epochs (ablation, not the submitted model)

First full training run, kept only to document the effect of training length:
test mean IoU 0.716 vs 0.750 after 15 epochs (`../outputs_15epochs/`), with every class except Skirt and Dress improving.
Same split, same protocol, same hardware. The submitted model and report use `outputs_15epochs/`.
