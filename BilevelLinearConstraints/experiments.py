import numpy as np
import importlib, copy, argparse
import matplotlib.pyplot as plt
import optuna
from problems import cvx_prob


#  Parse input
def parse_input():

    parser = argparse.ArgumentParser() 

    # General
    parser.add_argument('--out_iter', type=int, default=100)
    parser.add_argument('--prob', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--num_trials', type=int, default=30)
    parser.add_argument('--metrics', nargs='+', default=['loss'])
    parser.add_argument('--metric_score', type=str, default='loss')
    
    # [S]SIGD
    parser.add_argument('--ssigd_outer_iter', type=int, default=100)
    parser.add_argument('--ssigd_out_step_l', type=float, default=1e-3)
    parser.add_argument('--ssigd_out_step_u', type=float, default=1e-1)

    # DS-BLO
    parser.add_argument('--dsblo_outer_iter', type=int, default=100)
    parser.add_argument('--dsblo_gamma1_l', type=float, default=0.1)
    parser.add_argument('--dsblo_gamma1_u', type=float, default=10)
    parser.add_argument('--dsblo_gamma2_l', type=float, default=0.1)
    parser.add_argument('--dsblo_gamma2_u', type=float, default=10)
    parser.add_argument('--dsblo_beta_l', type=int, default=0.01)
    parser.add_argument('--dsblo_beta_u', type=int, default=1)
    
    args = parser.parse_args()

    return args


def hyperopt(args, prob, algs):
    
    init = exp_init(args.num_runs, prob)

    alg_stats = {}
    alg_res = {}
    for alg in algs:
        print(f'alg={alg}')
        study = optuna.create_study(direction='minimize')
        best_results = hyperopt_alg(args, init, study, prob, alg)

        alg_stats[alg] = best_results['alg_stats']
        alg_res[alg] = best_results['alg_res']
            
    plot_results_iter(algs, alg_stats, args.metrics, args.out_iter)
    plot_results_time(algs, alg_res, args.metrics, args.out_iter)


def hyperopt_alg(args, init, study, prob, alg):

    run_trial = getattr(importlib.import_module('experiments'), "run_trial_" + alg)

    best_results = {}

    for i in range(args.num_trials):
        try:
            study.optimize(lambda trial: run_trial(args, trial, init, prob, args.metrics, args.metric_score), n_trials=1)
        except Exception as e:
            continue

        trial_num = study.trials[-1].user_attrs['trial_num']

        # Best parameters
        best_results = {}
        best_results['trial_num'] = study.best_trial.number
        best_results['params'], best_results['value'] = study.best_params, study.best_value
        best_results['alg_stats'] = study.trials[study.best_trial.number].user_attrs['alg_stats']
        best_results['alg_res'] = study.trials[study.best_trial.number].user_attrs['alg_res']
        print(f"best_params: {best_results['params']}")
        print(f"best_value: {best_results['value']}")

        # curr results
        cur_results = {}
        cur_results['trial_num'] = trial_num
        cur_results['params'], cur_results['value'] = study.trials[-1].params, study.trials[-1].value

    return best_results


def score_fun(out_iter, alg_stats, metric):
    return np.mean(np.array(alg_stats[metric]['avg'][-int(round(out_iter/4)):]))


def run_trial_DsBlo(args, trial, init, prob, metrics, metric_score):

    # DS-BLO
    dsblo_out_iter = args.dsblo_outer_iter
    dsblo_gamma1 = trial.suggest_float("dsblo_gamma1", args.dsblo_gamma1_l, args.dsblo_gamma1_u)
    dsblo_gamma2 = trial.suggest_float("dsblo_gamma2", args.dsblo_gamma2_l, args.dsblo_gamma2_u)
    dsblo_beta = trial.suggest_float("dsblo_beta", args.dsblo_beta_l, args.dsblo_beta_u)
    param_dsblo = (prob, dsblo_out_iter, dsblo_gamma1, dsblo_gamma2, dsblo_beta)

    print(f"TRIAL NUMBER: {trial.number}")

    alg_res = run_alg('DsBlo', param_dsblo, init, args.num_runs)
    alg_stats = compute_stats(metrics, args.num_runs, alg_res)
    
    trial.set_user_attr('dsblo_out_iter', dsblo_out_iter)
    trial.set_user_attr('dsblo_gamma1', dsblo_gamma1)
    trial.set_user_attr('dsblo_gamma2', dsblo_gamma2)
    trial.set_user_attr('dsblo_beta', dsblo_beta)

    trial.set_user_attr('alg_res', alg_res)
    trial.set_user_attr('alg_stats', alg_stats)
    trial.set_user_attr('trial_num', trial.number)

    trial_score = score_fun(args.dsblo_outer_iter, alg_stats, metric_score)

    return trial_score


def run_trial_Ssigd(args, trial, init, prob, metrics, metric_score):

    # S[SIGD]
    ssigd_out_iter = args.ssigd_outer_iter
    ssigd_out_step = trial.suggest_float("ssigd_out_step", args.ssigd_out_step_l, args.ssigd_out_step_u)
    param_ssigd = (prob, ssigd_out_iter, ssigd_out_step)

    print(f"TRIAL NUMBER: {trial.number}")

    alg_res = run_alg('Ssigd', param_ssigd, init, args.num_runs)
    alg_stats = compute_stats(metrics, args.num_runs, alg_res)
    
    trial.set_user_attr('ssigd_out_iter', ssigd_out_iter)
    trial.set_user_attr('ssigd_out_step', ssigd_out_step)
    trial.set_user_attr('alg_res', alg_res)
    trial.set_user_attr('alg_stats', alg_stats)
    trial.set_user_attr('trial_num', trial.number)

    trial_score = score_fun(args.ssigd_outer_iter, alg_stats, metric_score)

    return trial_score


# Create initialization points for the different runs
def exp_init(num_runs, prob):
    init = {'x': {}, 'y': {}}
    for r in range(num_runs):
        init['x'][r] = np.expand_dims(np.random.rand(prob.x_dim), axis=1).astype(np.float32)
        init['y'][r] = prob.projy(np.expand_dims(np.random.rand(prob.y_dim), axis=1)).astype(np.float32)
    
    return init


# Run a single algorithm across all runs
def run_alg(alg, param, init, num_runs):

    alg_res = {}
    alg_inst = getattr(importlib.import_module('algorithms'), alg)(*param)

    for r in range(num_runs):
        print(f'algorithm={alg}, run={r}')
        x, y = copy.deepcopy(init['x'][r]), copy.deepcopy(init['y'][r])

        alg_res[r] = {}
        getattr(alg_inst, 'run')(copy.deepcopy(x),copy.deepcopy(y))

        alg_res[r]['x_iter'] = alg_inst.x_iter
        alg_res[r]['y_iter'] = alg_inst.y_iter
        alg_res[r]['gradF'] = alg_inst.gradF
        alg_res[r]['loss'] = alg_inst.loss
        alg_res[r]['iter_time'] = alg_inst.iter_time
        alg_res[r]['total_time'] = [alg_inst.iter_time[-1]]
        alg_inst.reset_eval()

    return alg_res


def compute_stats(metrics, num_runs, alg_res):

    alg_stats = {}

    for metric in metrics:
        alg_stats[metric]= {}
        metric_tmp = []

        for r in range(num_runs):
            metric_tmp.append(alg_res[r][metric])

        arr = np.array(metric_tmp)
        alg_stats[metric]['min'] = np.min(arr, axis=0)
        alg_stats[metric]['max'] = np.max(arr, axis=0)
        alg_stats[metric]['avg'] = np.mean(arr, axis=0)

    return alg_stats


def alg_names(alg):
    if alg == 'Ssigd':
        return '[S]SIGD'
    elif alg == 'DsBlo':
        return 'DS-BLO (ours)'


def plot_results_iter(algs, alg_stats, metrics, out_iter):

    for metric in metrics:
        fig = plt.figure()
        
        colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w'
        for alg, color in zip(algs, colors):
            iters = list(np.arange(len(alg_stats[alg][metric]['avg'])))
            plt.plot(iters, alg_stats[alg][metric]['avg'], label=alg_names(alg), color=color, linewidth=3.0)
            plt.fill_between(iters, alg_stats[alg][metric]['min'], alg_stats[alg][metric]['max'], color='green', alpha=0.3)

        # plt.style.use('seaborn-darkgrid')
        plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=16)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=16)  # fontsize of the tick labels
        plt.rc('legend', fontsize=14)  # legend fontsize

        plt.xlabel('iterations')
        plt.ylabel(f'{metric}')
        plt.legend(loc="upper right")
        plt.tight_layout()

        plt.savefig(f"{metric}_iter.pdf")
        # plt.show()


def plot_results_time(algs, alg_res, metrics, out_iter):

    for metric in metrics:
        fig = plt.figure()
        
        colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w'
        for alg, color in zip(algs, colors):
            plt.plot(alg_res[alg][0]['iter_time'], alg_res[alg][0][metric], label=alg_names(alg), color=color, linewidth=4.0)

        # plt.style.use('seaborn-darkgrid')
        plt.rc('axes', labelsize=24)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=20)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=20)  # fontsize of the tick labels
        plt.rc('legend', fontsize=20)  # legend fontsize

        plt.xlabel('time (s)')
        plt.ylabel(r'$F(x)$')
        plt.legend(loc="upper right")
        plt.tight_layout()

        plt.savefig(f"{metric}_time.pdf")


if __name__ == "__main__":

    args = parse_input()
    prob = cvx_prob()
    algs = ['Ssigd', 'DsBlo']

    hyperopt(args, prob, algs)

    