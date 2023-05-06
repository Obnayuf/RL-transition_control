import argparse
import time
import os

import env_wrapper
import algo_wrapper

if __name__ == '__main__':
    os.environ['DISPLAY'] = ':1'
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--algo',
        type=str,
        default='CPO',
        help='Choose from: {PolicyGradient, PPO, PPOLag, NaturalPG, TRPO, TRPOLag, PDO, NPGLag, CPO, PCPO, FOCOPS, CPPOPid',
    )
    parser.add_argument(
        '--env-id',
        type=str,
        default='BasicEnv-v0',
        help='The name of test environment',
    )
    parser.add_argument(
        '--parallel', default=1, type=int, help='Number of paralleled progress for calculations.'
    )

    args, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [eval(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    env = env_wrapper.EnvWrapper(args.env_id)
    agent = algo_wrapper.AlgoWrapper(args.algo, env, parallel=args.parallel, custom_cfgs=unparsed_dict)
    agent.learn()
