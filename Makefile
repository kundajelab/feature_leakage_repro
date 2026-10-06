# Convenience targets for reproducing the Feature-Leakage figures.
#
# Set the GPU and (optionally) the akundaje group root:
#   make fig3 GPU=0
#   make fig2 GPU=1 AKUNDAJE=/oak/stanford/groups/akundaje
#
# Each target activates the right conda env via `conda run`.

GPU ?= 0
AKUNDAJE ?= /oak/stanford/groups/akundaje
DL  = CUDA_VISIBLE_DEVICES=$(GPU) FEATLEAK_AKUNDAJE_ROOT=$(AKUNDAJE) conda run -n featleak-dl python
AG  = CUDA_VISIBLE_DEVICES=$(GPU) FEATLEAK_AKUNDAJE_ROOT=$(AKUNDAJE) conda run -n featleak-alphagenome python
AGT = CUDA_VISIBLE_DEVICES=$(GPU) FEATLEAK_AKUNDAJE_ROOT=$(AKUNDAJE) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True conda run -n featleak-agtorch python

.PHONY: check fig1 fig2 fig3 fig4 fig5 fig6 fig7 fig1-heavy fig2-heavy

check:            ## verify all data paths resolve
	conda run -n featleak-dl python -m common.config

fig1:             ## Fig 1 E-H (Beluga, four estimators x four quartiles)
	cd figures/figure1 && for m in OCT-SOX AP-1 HNF4 CTCF; do \
	  $(DL) run_beluga_grid.py --motif $$m --hit-set genome --n-per-quartile 3000 --tag _big; done
	cd figures/figure1 && $(DL) plot_beluga_grid.py --hit-set genome --file-tag _big --band ci95 --mark-sig

fig1-heavy:       ## Fig 1 C (AlphaGenome 2 kb vs ChromBPNet endogenous ablation)
	cd figures/figure1 && $(DL) run_chrombpnet_logfc.py
	cd figures/figure1 && $(AGT) run_alphagenome_torch.py --seq-len 2048 --max-loci 400 --out-suffix _2kb
	cd figures/figure1 && $(DL) run_logfc_styled.py

fig2:             ## Fig 2 B + Supp 2 (ChromBPNet folds, Enformer, Borzoi; plots)
	cd figures/figure2 && $(DL) run_raw_chrombpnet.py
	cd figures/figure2 && $(DL) run_raw_grelu.py --model enformer
	cd figures/figure2 && $(DL) run_raw_grelu.py --model borzoi
	cd figures/figure2 && $(DL) plot_raw_insertion.py

fig2-heavy:       ## Fig 2 B AlphaGenome at 524 kb (run before `make fig2` plots)
	cd figures/figure2 && $(AGT) run_raw_alphagenome.py

fig3:             ## Fig 3 (AP1 counting on frozen Beluga)
	cd figures/figure3 && $(DL) run.py

fig4:             ## Fig 4 (single vs multi-task) + MoDISco
	cd figures/figure4 && $(DL) run_experiment.py
	cd figures/figure4 && $(DL) run_modisco.py

fig5:             ## Fig 5 (simulation + entanglement)
	cd figures/figure5 && $(DL) run_simulation.py   # defaults (n=10000, seed=1) reproduce Fig 5b
	cd figures/figure5 && $(DL) run_entanglement.py

fig6:             ## Fig 6 (design leakage; plots from precomputed results)
	cd figures/figure6 && $(DL) run_plots.py

fig7:             ## Fig 7 (ChromBPNet-gated AlphaGenome fine-tuning)
	cd experiments/ag_deleak_finetune && $(AGT) finetune_dermis_torch.py --stage cache
	cd experiments/ag_deleak_finetune && $(DL) finetune_dermis_torch.py --stage gate
	cd experiments/ag_deleak_finetune && $(AGT) finetune_dermis_torch.py --stage train
	cd experiments/ag_deleak_finetune && $(DL) finetune_dermis_torch.py --stage obs
	cd experiments/ag_deleak_finetune && $(AGT) finetune_dermis_torch.py --stage eval
	cd experiments/ag_deleak_finetune && $(AGT) finetune_dermis_torch.py --stage ism --ism-locus chr9:127021880
	cd experiments/ag_deleak_finetune && $(DL) plot_fig7_dermis.py
