import argparse
import os
import time
import warnings
from datetime import datetime
from ray.rllib.algorithms.ppo import PPOConfig

# import gymnasium as gym # for env Ray version: 2.4.0
import ray
from ray import tune
# from ray.rllib.agents.ppo import PPOTrainer  # Ray version: 2.0.0
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPO  # for env Ray version: 2.4.0
from ray.rllib.models import ModelCatalog
from tqdm import tqdm

from Environments import TornadoCashGameEnvEvader
from Models import GameNormModel

warnings.filterwarnings("ignore")


class RunGame:
    @staticmethod
    def load_ppo_trainer(config):
        trainer = PPO(config=config)  # for env Ray version: 2.4.0
        # trainer = PPOTrainer(config=config)  # for env Ray version: 2.0.0
        return trainer

    @staticmethod
    def run_parallel(no_iter_in, config):
        tune.run(
            "PPO",  # Use the PPO algorithm
            config=config,
            stop={"training_iteration": no_iter_in},  # Stop after a certain number of steps
            verbose=1,  # Set the verbosity level
        )

    @staticmethod
    def run_iteratively(no_iter_in, log_dir_in, data_log_dir_in, checkpoint_dir, config=None):
        os.environ["TMPDIR"] = '<PATH-TO-TMPDIR>'
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"
        ray.init()
        trainer = RunGame.load_ppo_trainer(config=config)

        result_file = open(f'{log_dir_in}/dryrun.csv', 'a')
        result_file.write("episode_reward_mean,episode_reward_max,episode_reward_min\n")
        result_file.flush()

        for _i in tqdm(range(no_iter_in)):
            train = trainer.train()

            trainer.save(checkpoint_dir)

            episode_reward_mean = train['episode_reward_mean']
            episode_reward_max = train['episode_reward_max']
            episode_reward_min = train['episode_reward_min']
            print(
                f'episode_reward_mean :{episode_reward_mean}\tepisode_reward_max :{episode_reward_max}\t'
                f'episode_reward_min :{episode_reward_min}')
            result_file.write(f'{episode_reward_mean},{episode_reward_max},{episode_reward_min}\n')
            result_file.flush()

        result_file.flush()
        result_file.close()

        ray.shutdown()


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='name of the experiment')
    parser.add_argument('--data_dir', type=str, help='dir for the chain history data')
    parser.add_argument('--log_dir', type=str, help='dir for the log training logs')
    parser.add_argument('--checkpoint_dir', type=str, help='dir for the model checkpoints')
    parser.add_argument('--no_addresses_agent_challenge_table', type=int, help='no_addresses_agent_challenge_table')
    parser.add_argument('--agent_challenge_table', type=str, help='agent_challenge_table')
    parser.add_argument('--agent_address_range_starts', type=int, help='agent_address_range_starts')
    parser.add_argument('--agent_address_range_end', type=int, help='agent_address_range_end')
    parser.add_argument('--agent_mutable_address_range_start', type=int, help='agent_mutable_address_range_start')
    parser.add_argument('--agent_mutable_address_range_end', type=int, help='agent_mutable_address_range_end')
    parser.add_argument('--crowd_address_range_starts', type=int, help='crowd_address_range_starts')
    parser.add_argument('--no_of_crowd', type=int, help='no_of_crowd')
    parser.add_argument('--no_of_wallets_for_each_crowd_agent', type=int, help='no_of_wallets_for_each_crowd_agent')
    parser.add_argument('--amount_of_money_in_each_crowd', type=int, help='amount_of_money_in_each_crowd')
    parser.add_argument('--fcnet_hiddens', type=str, help='fcnet_hiddens')
    parser.add_argument('--no_iter', type=int, help='no_iter')

    args = parser.parse_args()

    TEST_NAME = args.name
    EVADER_DATA_LOG_DIR = args.data_dir
    EVADER_LOG_DIR = args.log_dir
    EVADER_CHECK_POINT_DIR = args.checkpoint_dir
    no_addresses_agent_challenge_table = args.no_addresses_agent_challenge_table
    agent_challenge_table = eval(args.agent_challenge_table)
    agent_address_range_starts = args.agent_address_range_starts
    agent_address_range_end = args.agent_address_range_end
    agent_mutable_address_range_start = args.agent_mutable_address_range_start
    agent_mutable_address_range_end = args.agent_mutable_address_range_end
    crowd_address_range_starts = args.crowd_address_range_starts
    no_of_crowd = args.no_of_crowd
    no_of_wallets_for_each_crowd_agent = args.no_of_wallets_for_each_crowd_agent
    amount_of_money_in_each_crowd = args.amount_of_money_in_each_crowd
    fcnet_hiddens = eval(args.fcnet_hiddens)
    no_iter = args.no_iter

    current_datetime = datetime.now()
    folder_name = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    data_log_dir = f'{EVADER_DATA_LOG_DIR}{TEST_NAME}{folder_name}'
    log_dir = f'{EVADER_LOG_DIR}{TEST_NAME}{folder_name}'
    check_point_dir = f'{EVADER_CHECK_POINT_DIR}{TEST_NAME}{folder_name}'
    os.mkdir(data_log_dir)
    os.mkdir(log_dir)
    os.mkdir(check_point_dir)
    ModelCatalog.register_custom_model("model_with_batch_normalization", GameNormModel)

    _env_params = {
        'block_size': 5,
        'max_wait_time': 5,
        'no_addresses_agent_challenge_table': no_addresses_agent_challenge_table,
        'agent_challenge_table': agent_challenge_table,
        'agent_address_range_starts': agent_address_range_starts,
        'agent_address_range_end': agent_address_range_end,
        'agent_mutable_address_range_start': agent_mutable_address_range_start,
        'agent_mutable_address_range_end': agent_mutable_address_range_end,
        'crowd_address_range_starts': crowd_address_range_starts,
        'no_of_crowd': no_of_crowd,
        'no_of_wallets_for_each_crowd_agent': no_of_wallets_for_each_crowd_agent,
        'amount_of_money_in_each_crowd': amount_of_money_in_each_crowd
    }

    _config = (
        PPOConfig()
        .environment(
            env=TornadoCashGameEnvEvader,
            env_config=_env_params
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=1,
            horizon=10000
        )
        .training(
            model={"fcnet_hiddens": fcnet_hiddens}
        )
        .callbacks(DefaultCallbacks)
        .to_dict()
    )

    RunGame.run_iteratively(no_iter_in=no_iter, log_dir_in=log_dir, data_log_dir_in=data_log_dir,
                            checkpoint_dir=check_point_dir, config=_config)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time taken: {execution_time} seconds")
    # End!!
