SOLUTION.md
Final Solution Description
Modified files:
zo_optimizer.py – implements zero‑order optimisation using repeated SPSA estimates.
head_init.py – Kaiming uniform initialisation with small scale (0.1).
augmentation.py – includes AutoAugment, RandomCrop, ColorJitter, RandomErasing, and GaussianNoise.
train_data.py – uses persistent workers and increased num_workers.
Key choices:
- Repeated SPSA - instead of a single gradient estimate per step, we average 16 independent SPSA estimates (each using 2 forward passes). This reduces gradient noise and makes the update more reliable.
- Learning only the classification head (fc.weight, fc.bias). The head contains only ~51k parameters, which is manageable for zero‑order optimisation within the budget.
- Small learning rate (0.01) and small perturbation scale (0.001) – prevent divergence.
- No momentum or adaptive methods – kept simple to avoid additional hyperparameters.
What contributed most to improvement – the use of multiple gradient estimates per step (repeated sampling) was the only technique that gave any measurable improvement over the initialised head.

Experiments and Failed Attempts
We systematically explored:
1) Full fine‑tuning of all layers – led to NaN loss or no progress, due to the extremely large parameter space (11M).
2) Fine‑tuning the last residual block + head – still too many parameters (~1.5M), no convergence.
3) SPSA with momentum, RMSprop, weight decay, gradient clipping – none improved accuracy.
3) Decaying learning rate and epsilon (standard SPSA schedule) – did not help.
4) ZO‑Muon orthogonalisation – after fixing memory issues, still gave no improvement (accuracy remained ~1%).
5) Repeated SPSA with 16 estimates – yielded a slight but consistent improvement (1.39%).
6) Repeated SPSA with 64 estimates – theoretically better, but training time became impractical (>20 hours).

To reproduce current best solution, run:
python validate.py --data_dir ./data --batch_size 32 --n_batches 256 --output results.json 

It iakes about 7 hours
Conclusion: Within the strict budget of 8192 forward passes, zero‑order optimisation can only marginally improve the randomly initialised head. The main limitation is the high variance of gradient estimates, which cannot be fully compensated by repeated sampling without exceeding the budget.
