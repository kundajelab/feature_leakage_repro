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

.PHONY: check fig1 fig2 fig3 fig4 fig5 fig6 fig1-heavy fig2-heavy

check:            ## verify all data paths resolve
	conda run -n featleak-dl python -m common.config

fig1:             ## Fig 1 E-H (Beluga), fast
	cd figures/figure1 && $(DL) run_beluga.py --method ablation

fig1-heavy:       ## Fig 1 Borzoi/Enformer (gReLU) + AlphaGenome (local)
	cd figures/figure1 && $(DL) run_grelu.py --model enformer
	cd figures/figure1 && $(AG) run_alphagenome.py

fig2:             ## Fig 2 B,F (ChromBPNet)
	cd figures/figure2 && $(DL) run_chrombpnet.py

fig2-heavy:       ## Fig 2 C,E (AlphaGenome local)
	cd figures/figure2 && $(AG) run_alphagenome.py

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
