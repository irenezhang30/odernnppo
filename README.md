modified from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail



## Requirements

* Python 3 (it might work with Python 2, but I didn't test it)
* [PyTorch](http://pytorch.org/)
* [OpenAI baselines](https://github.com/openai/baselines)

In order to install requirements, follow:

```bash
# PyTorch
conda install pytorch torchvision -c soumith

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# Other requirements
pip install -r requirements.txt
```

# Running experiments:
* ODE-RNN-PPO:
python3 main.py --env-name "acrobot" --algo ppo --use-gae --lr 3e-4 --clip-param 0.2 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 1 --log-interval 1 --entropy-coef 0.01 --use-ode --recurrent-policy

* RNN-PPO (baseline):
python3 main.py --env-name "acrobot" --algo ppo --use-gae --lr 3e-4 --clip-param 0.2 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 1 --log-interval 1 --entropy-coef 0.01 --recurrent-policy
