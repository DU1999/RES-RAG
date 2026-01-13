import argparse

def TrainingHyperparameters():
    parser = argparse.ArgumentParser(description="RES-RAG training parameters")

    parser.add_argument('--generator_api', type=str, required=True, help='Generator API key.')
    parser.add_argument('--generator_url', type=str, required=True, help='Generator base url.')
    parser.add_argument('--generator_model', type=str, default="llama-3.1-8b-instruct", required=True, help='Generator model.')
    parser.add_argument('--generator_tokens', type=float, default=512, required=True, help='Generator max tokens.')
    parser.add_argument('--generator_temperature', type=float, default=0.2, required=True, help='Generator temperature.')
    parser.add_argument('--generator_topp', type=float, default=0.1, required=True, help='Generator topp.')
    parser.add_argument('--generator_workers', type=int, default=8, required=True, help='Number of concurrent threads.')
    parser.add_argument('--generator_retries', type=int, default=3, required=True, help='Number of concurrent requests.')
    parser.add_argument('--generator_sleep', type=float, default=0.5, required=True, help='Concurrent waiting time.')
    
    
    parser.add_argument('--alpha_f1', type=float, default=2.0, required=True, help='Reward coefficient.')
    parser.add_argument('--alpha_ret', type=float, default=1.0, required=True, help='Retrieval consistency coefficient.')
    
    
    parser.add_argument('--bad_answer_penalty', type=float, default=-2.0, required=True, help='Penalty for poor samples.')
    parser.add_argument('--sys_fail_penalty', type=float, default=-5.0, required=True, help='System failure penalty.')
    
    
    parser.add_argument('--scale', type=float, default=1.0, required=True, help='Scaling factor.')
    parser.add_argument('--clip_min', type=float, default=-5.0, required=True, help='Minimum cropping.')
    parser.add_argument('--clip_max', type=float, default=5.0, required=True, help='Maximum cropping.')
    

    return parser.parse_args()

