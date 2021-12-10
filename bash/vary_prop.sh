algo="PPO"
# shifts=("task" "background" "random" "blackout")
shifts=("blackout")
probs=(0 0.01 0.05 0.1 0.2 0.5 0.8 1.0)
std_dev=10

for shift in "${shifts[@]}"
do
    wandb_project_name="state-vary-outlier-prop-$shift"
    for prob in "${probs[@]}"
    do
        printf "Running for $algo, $shift, $prob\n";
        cmd="nohup python -u train.py env=CartPole-v1 hyperparams.PPO.CartPole_v1.n_timesteps=100000 model=$algo image_input=False ood_config.use=True ood_config.prob=$prob ood_config.type=$shift ood_config.random_std=$std_dev wandb.project=$wandb_project_name > ${algo}_${shift}_${prob}.out &"
        echo "$cmd"
        eval "$cmd"
    done
done