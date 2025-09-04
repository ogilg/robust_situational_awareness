# Test steering with projection-equivalent language vectors

# Baseline - no steering (coeff 0)
# SA_SPLIT=test python -m experiments.benchmark_with_steering --model qwen-2.5-14b-instruct --n 200 --examples 0 --tasks id_leverage stages output_control --vector-file experiments/vectors/steering_vectors_qwen-2.5-14b-instruct__sp_diff_proj_equiv.npz --steering-mode add --coefficient 0.0 --layers 25 --out-dir experiments/results/language_steering --comment "baseline_coeff_0"

#SA_SPLIT=test python -m experiments.benchmark_with_steering --model qwen-2.5-14b-instruct --n 200 --examples 0 --tasks id_leverage --vector-file experiments/vectors/steering_vectors_qwen-2.5-14b-instruct__sp_diff_proj_equiv.npz --steering-mode add --coefficient 1.0 --layers 25 --out-dir experiments/results/language_steering --comment "sp_proj_equiv_coeff_2.5"

# SP variant projection-equivalent - coeff 0.25
SA_SPLIT=test python -m experiments.benchmark_with_steering --model qwen-2.5-14b-instruct --n 200 --examples 0 --tasks id_leverage --vector-file experiments/vectors/steering_vectors_qwen-2.5-14b-instruct__sp_diff_proj_equiv.npz --steering-mode add --coefficient 5.0# --layers 25 --out-dir experiments/results/language_steering --comment "sp_proj_equiv_coeff_2.5"

SA_SPLIT=test python -m experiments.benchmark_with_steering --model qwen-2.5-14b-instruct --n 200 --examples 0 --tasks id_leverage --vector-file experiments/vectors/steering_vectors_qwen-2.5-14b-instruct__sp_diff_proj_2.0x.npz --steering-mode add --coefficient 2.5 --layers 25 --out-dir experiments/results/language_steering --comment "sp_proj_2x_coeff_2.5"


SA_SPLIT=test python -m experiments.benchmark_with_steering --model qwen-2.5-14b-instruct --n 200 --examples 0 --tasks id_leverage --vector-file experiments/vectors/steering_vectors_qwen-2.5-14b-instruct__plain_diff_proj_equiv.npz --steering-mode add --coefficient 1.0 --layers 25 --out-dir experiments/results/language_steering --comment "plain_proj_equiv_coeff_1.0"

# Plain variant projection-equivalent - coeff 0.25  
SA_SPLIT=test python -m experiments.benchmark_with_steering --model qwen-2.5-14b-instruct --n 200 --examples 0 --tasks id_leverage --vector-file experiments/vectors/steering_vectors_qwen-2.5-14b-instruct__plain_diff_proj_equiv.npz --steering-mode add --coefficient 2.5 --layers 25 --out-dir experiments/results/language_steering --comment "plain_proj_equiv_coeff_2.5"

# Plain variant projection-equivalent - coeff 0.5
SA_SPLIT=test python -m experiments.benchmark_with_steering --model qwen-2.5-14b-instruct --n 200 --examples 0 --tasks id_leverage --vector-file experiments/vectors/steering_vectors_qwen-2.5-14b-instruct__plain_diff_proj_2.0x.npz --steering-mode add --coefficient 2.5 --layers 25 --out-dir experiments/results/language_steering --comment "plain_proj_2x_coeff_2.5"
