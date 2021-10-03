import argparse
import os

import torch.multiprocessing as mp

from src.nli.training import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="If set, we only use CPU.")
    parser.add_argument("--single_gpu", action="store_true", help="If set, we only use single GPU.")
    parser.add_argument("--fp16", action="store_true", help="If set, we will use fp16.")

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )

    # environment arguments
    parser.add_argument('-s', '--seed', default=1, type=int, metavar='N',
                        help='manual random seed')
    parser.add_argument('-n', '--num_nodes', default=1, type=int, metavar='N',
                        help='number of nodes')
    parser.add_argument('-g', '--gpus_per_node', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--node_rank', default=0, type=int,
                        help='ranking within the nodes')

    # experiments specific arguments
    parser.add_argument('--debug_mode',
                        action='store_true',
                        dest='debug_mode',
                        help='weather this is debug mode or normal')

    parser.add_argument(
        "--model_class_name",
        type=str,
        help="Set the model class of the experiment.",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Set the name of the experiment. [model_name]/[data]/[task]/[other]",
    )

    parser.add_argument(
        "--save_prediction",
        action='store_true',
        dest='save_prediction',
        help='Do we want to save prediction')

    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="If we want to resume model training, we need to set the resume path to restore state dicts.",
    )
    parser.add_argument(
        "--global_iteration",
        type=int,
        default=0,
        help="This argument is only used if we resume model training.",
    )

    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--total_step', default=-1, type=int, metavar='N',
                        help='number of step to update, default calculate with total data size.'
                             'if we set this step, then epochs will be 100 to run forever.')

    parser.add_argument('--sampler_seed', default=-1, type=int, metavar='N',
                        help='The seed the controls the data sampling order.')

    parser.add_argument(
        "--per_gpu_train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=64, type=int, help="Batch size per GPU/CPU for evaluation.",
    )

    parser.add_argument("--max_length", default=160, type=int, help="Max length of the sequences.")

    parser.add_argument("--warmup_steps", default=-1, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    parser.add_argument(
        "--eval_frequency", default=1000, type=int, help="set the evaluation frequency, evaluate every X global step.",
    )

    parser.add_argument("--train_data",
                        type=str,
                        help="The training data used in the experiments.")

    parser.add_argument("--train_weights",
                        type=str,
                        help="The training data weights used in the experiments.")

    parser.add_argument("--eval_data",
                        type=str,
                        help="The training data used in the experiments.")

    args = parser.parse_args()

    if args.cpu:
        args.world_size = 1
        train(-1, args)
    elif args.single_gpu:
        args.world_size = 1
        train(0, args)
    else:  # distributed multiGPU training
        #########################################################
        args.world_size = args.gpus_per_node * args.num_nodes  #
        # os.environ['MASTER_ADDR'] = '152.2.142.184'  # This is the IP address for nlp5
        # maybe we will automatically retrieve the IP later.
        os.environ['MASTER_PORT'] = '88888'  #
        mp.spawn(train, nprocs=args.gpus_per_node, args=(args,))  # spawn how many process in this node
        # remember train is called as train(i, args).
        #########################################################

if __name__ == '__main__':
    main()