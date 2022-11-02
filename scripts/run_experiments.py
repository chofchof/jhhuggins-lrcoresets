from __future__ import print_function

import sys
import os
import argparse
import itertools

import numpy as np

from coresets.algorithms import (construct_lr_coreset_with_kmeans,
                                 random_data_subset,
                                 full_data)
from coresets.data import (generate_binary_data,
                           generate_reverse_mixture)
from coresets.inference import mh
from coresets.evaluation import (mean_relative_squared_error,
                                 log_likelihood,
                                 prediction_error,
                                 make_weighted_brier_score,
                                 make_polynomial_mmd_evaluation,
                                 median_mean_error, mean_mean_error,
                                 median_variance_error, mean_variance_error)
from coresets.pipeline import (Experiment,
                               plot_subsample,
                               plot_results,
                               plot_means_and_vars)
from coresets.utils import (create_folder_if_not_exist,
                            pretty_file_string_from_dict)

DS_FORMAT = 'NAME:TRAIN_PATH:TEST_PATH:FILE_FORMAT'

CHEMREACT_SPEC = 'ChemReact:data/ds1.100_train.npz:data/ds1.100_test.npz:npz'
COVTYPE_SPEC = 'CovType:data/covtype_train.svm:data/covtype_test.svm:svm'
WEBSPAM_SPEC = 'Webspam:data/webspam_train.svm:data/webspam_test.svm:svm'
SYNTHETIC_NAMES = ['Binary',
                   'Mixture',
                   'MixtureUnbalanced',
                   ]
SYNTHETIC_SPEC_STUB = '{0}:data/synthetic/{1}:data/synthetic/{1}_test:synthetic'
SYNTHETIC_SPECS = [SYNTHETIC_SPEC_STUB.format(name, name.lower())
                        for name in SYNTHETIC_NAMES]
RESULTS_DIR = 'results'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', metavar='dataset', nargs='*',
                        action='append',
                        help='data sets in the format ' + DS_FORMAT)
    parser.add_argument('--chemreact', dest='datasets', action='append_const',
                        const=[CHEMREACT_SPEC],
                        help='include ChemReact dataset')
    parser.add_argument('--covtype', dest='datasets', action='append_const',
                        const=[COVTYPE_SPEC], help='include CovType dataset')
    parser.add_argument('--webspam', dest='datasets', action='append_const',
                        const=[WEBSPAM_SPEC], help='include Webspam dataset')
    parser.add_argument('--synth-bin', dest='datasets', action='append_const',
                        const=[SYNTHETIC_SPECS[0]])
    parser.add_argument('--synth-mix', dest='datasets', action='append_const',
                        const=[SYNTHETIC_SPECS[1]])
    parser.add_argument('--synth-mix-unbalanced', dest='datasets',
                        action='append_const', const=[SYNTHETIC_SPECS[2]])
    parser.add_argument('-s', '--subsample-sizes', metavar='size', nargs='*',
                        type=int, default=[10000])
    parser.add_argument('-d', '--max-dimension', type=int, default=0)
    parser.add_argument('-i', '--iters', type=int, default=-5000)
    parser.add_argument('--include-offset', action='store_true',
                        help='add dummy feature equal to 1')
    parser.add_argument('-k', '--num-clusters', nargs='*', type=int,
                        default=[4])
    parser.add_argument('-R', '--radii', metavar='radius', nargs='*',
                        type=float, default=[1.0])
    parser.add_argument('-r', '--repetitions', type=int, default=10)
    parser.add_argument('--mh-alg', default='MALA')
    parser.add_argument('--timing', action='store_true',
                        help='run a timing experiment')
    parser.add_argument('--skip-plot', action='store_true',
                        help='skip plotting the results')
    parser.add_argument('--skip-run', action='store_true',
                        help='skip running the experiments')
    return parser.parse_args()


def nice_str(o):
    if isinstance(o, list):
        return ','.join(map(nice_str, o))
    else:
        return str(o)


def experiment_name(args, datasets):
    ds_names = [ds[0].lower() for ds in datasets]
    ds_str = '-'.join(ds_names)
    attributes = [('subsample_sizes', 'Ms'),
                  ('radii', 'Rs'),
                  ('num_clusters', 'Ks'),
                  ('max_dimension', 'd'),
                  ]
    if args.iters > 0:
        attributes.append(('iters', 'iters'))
    attr_dict = { a[1] : nice_str(getattr(args, a[0])) for a in attributes }
    attr_str = pretty_file_string_from_dict(attr_dict)
    if args.timing:
        expr_str = 'timing-experiment'
    else:
        expr_str = 'experiment'
    name = '-'.join([ds_str, attr_str, expr_str, args.mh_alg])
    return name


def run_experiment(args, datasets, test_datasets):
    algs = [('Coreset', construct_lr_coreset_with_kmeans),
            ('Random', random_data_subset),
            ('Full', full_data)]
    parameter_dicts = []
    parameters = { 'kmeans_subsample_size' : 'auto'}
    for m in args.subsample_sizes:
        for k in args.num_clusters:
            for R in args.radii:
                parameters.update({'output_size_param': m,
                                   'K' : k,
                                   'R' : R,
                                  })
                parameter_dicts.append(parameters.copy())
    evals = [('MRSE', mean_relative_squared_error),
             ('MMD2', make_polynomial_mmd_evaluation(degree=2)),
             ('MMD3', make_polynomial_mmd_evaluation(degree=3)),
             ('MedMeanErr', median_mean_error),
             ('MeanMeanErr', mean_mean_error),
             ('MedVarErr', median_variance_error),
             ('MeanVarErr', mean_variance_error),
             ]
    if len(test_datasets) > 0:
        evals.extend([('TLL', log_likelihood),
                      ('TPE', prediction_error),
                      ('WBS-10', make_weighted_brier_score(10)),
                      ('WBS-100', make_weighted_brier_score(100)),
                      ])

    # create the experiment object; make sure the target_name is set to Full so
    # other algorithms are compared to it
    expt_name = experiment_name(args, datasets)
    experiment = Experiment(expt_name, algs, parameter_dicts, evals,
                            datasets, test_datasets, target_name='Full',
                            run_target_once=True)

    steps = args.iters if args.iters > 0 else -args.iters
    experiment.run(n_trials=args.repetitions, steps=steps, thin=5,
                   target_alg_steps=5*steps, mh_alg=args.mh_alg,
                   include_offset=args.include_offset)


def make_plots(args, datasets, test_datasets, show=False):
    expt_name = experiment_name(args, datasets)
    metrics = [('MMD2', 'Polynomial MMD 2', 'log', ['Full']),
               ('MMD3', 'Polynomial MMD 3', 'log', ['Full']),
               ('MRSE', 'MRSE', 'log', ['Full']),
               ('MedMeanErr', 'Median Mean Error', 'linear', ['Full']),
               ('MeanMeanErr', 'Mean Mean Error', 'linear', ['Full']),
               ('MedVarErr', 'Median Variance Error', 'linear', ['Full']),
               ('MeanVarErr', 'Mean Variance Error', 'linear', ['Full']),
               ]
    if len(test_datasets) > 0:
        metrics.extend([('TLL', 'Test Log-Likelihood', 'linear', []),
                        ('WBS-10', 'Weighted Brier Score (10)', 'linear', []),
                        ('WBS-100', 'Weighted Brier Score (100)', 'linear', []),
                        ])
    xsel = lambda r: [r.prms['output_size_param']]
    xlabel = 'Subset Size'
    tsel = lambda r: np.array(r.alg_times)+np.array(r.sample_times)
    tlabel = 'Total Runtime (s)'


    alg_to_color = {}
    data_to_color = {}
    for metric, metric_name, yscale, excluded_algs in metrics:
        ysel = lambda r : [ev[metric] for ev in r.evals]
        ylabel = metric_name
        #Y vs subset size
        alg_to_color, data_to_color = plot_results(
                     expt_name, x_selector=xsel, x_label=xlabel,
                     y_selector=ysel, y_label=ylabel, plot_type='line',
                     xscale='log', yscale=yscale, show=show,
                     alg_to_color=alg_to_color, data_to_color=data_to_color,
                     excluded_algs=excluded_algs)
        #Y vs time
        alg_to_color, data_to_color = plot_results(
                     expt_name, x_selector=tsel, x_label=tlabel,
                     y_selector=ysel, y_label=ylabel, plot_type='line',
                     xscale='log', yscale=yscale, show=show,
                     alg_to_color=alg_to_color, data_to_color=data_to_color,
                     excluded_algs=excluded_algs)


    selector = lambda r: r.prms.get('output_size_param')
    alg_to_color = plot_means_and_vars(expt_name, 'Full',
                                       'output_size_param', selector,
                                       alg_to_color)

    #Time scatter plots -- subset selection time vs coreset size
    alg_to_color, data_to_color = plot_results(
                     expt_name, x_selector=xsel, x_label=xlabel,
                     y_selector=lambda r: r.alg_times,
                     y_label='Subset Selection Time', plot_type='line',
                     xscale='log', yscale='log', show=show,
                     alg_to_color=alg_to_color, data_to_color=data_to_color,
                     excluded_algs=['Full'])

    ysel = lambda r: np.array(r.alg_times) / (np.array(r.alg_times) +
                                              np.array(r.sample_times))
    alg_to_color, data_to_color = plot_results(
                     expt_name, x_selector=xsel, x_label=xlabel,
                     y_selector=ysel, y_label='% time to select subset',
                     plot_type='line', xscale='log', yscale='log', show=show,
                     alg_to_color=alg_to_color, data_to_color=data_to_color,
                     excluded_algs=['Full'])


def run_timing_experiment(args, datasets, test_datasets):
    algs = [('Coreset', construct_lr_coreset_with_kmeans),
            ('Full', full_data)
            ]
    parameter_dicts = []
    parameters = { 'kmeans_subsample_size' : 'auto'}
    for m in args.subsample_sizes:
        for k in args.num_clusters:
            for R in args.radii:
                parameters.update({'output_size_param': m,
                                   'K' : k,
                                   'R' : R,
                                  })
                parameter_dicts.append(parameters.copy())

    evals = []
    # create the experiment object; make sure the target_name is set to Full so
    # other algorithms are compared to it
    expt_name = experiment_name(args, datasets)
    experiment = Experiment(expt_name, algs, parameter_dicts, evals,
                            datasets, test_datasets, target_name='Full',
                            run_target_once=True)

    steps = args.iters if args.iters > 0 else -args.iters
    experiment.run(n_trials=args.repetitions, steps=steps, target_alg_steps=2,
                   thin=10, mh_alg=args.mh_alg,
                   include_offset=args.include_offset)


def make_timing_plots(args, datasets, test_datasets, show=False):
    expt_name = experiment_name(args, datasets)
    legend_kwargs = { 'bbox_to_anchor' : (.5,.9),
                      'loc' : 3,
                      'ncol' : 2,
                      'mode' : 'expand',
                      'borderaxespad' : 0. }
    xsel = lambda r: [r.prms['output_size_param']]
    xlabel = 'Subset Size'

    alg_to_color = {}
    data_to_color = {}

    #Time scatter plots -- subset selection time vs coreset size
    ysel = lambda r: r.alg_times
    alg_to_color, data_to_color = plot_results(
                        expt_name, x_selector=xsel, x_label=xlabel,
                        y_selector=ysel,
                        y_label='Subset Selection Time (s)', plot_type='line',
                        xscale='log', yscale='log', legend_kwargs=legend_kwargs,
                        show=show, alg_to_color=alg_to_color,
                        data_to_color=data_to_color, excluded_algs=['Full'])

    ysel = lambda r: 100 * np.array(r.alg_times) / (np.array(r.alg_times) +
                                                    np.array(r.sample_times))
    alg_to_color, data_to_color = plot_results(
                         expt_name, x_selector=xsel, x_label=xlabel,
                         y_selector=ysel, y_label='% Time Creating Coreset',
                         plot_type='line', xscale='log', yscale='log',
                         legend_kwargs=legend_kwargs, show=show,
                         alg_to_color=alg_to_color, data_to_color=data_to_color,
                         excluded_algs=['Full'])


def construct_datasets(args):
    datasets = []
    test_datasets = []
    flat_datasets = list(itertools.chain(*args.datasets))
    for ds in flat_datasets:
        parts = ds.split(':')
        if len(parts) != 4:
            sys.exit('Invalid dataset format. Must be in the form ' + DS_FORMAT)
        dsname, train, test, ftype = parts
        train = os.path.join('..', train)
        if len(test) > 0:
            test = os.path.join('..', test)
        if ftype == 'synthetic':
            train, test, ftype = generate_synthetic_data(args, dsname,
                                                         train, test)
        datasets.append((dsname, train, ftype))
        if len(test) > 0:
            test_datasets.append((dsname, test, ftype))

    return datasets, test_datasets


def generate_synthetic_data(args, dsname, train, test):
    if dsname in SYNTHETIC_NAMES[1:]: # synth-rev or synth-rev-un
        args.include_offset = True
    else:
        args.include_offset = False
    DIM = 10 if args.max_dimension <= 0 else args.max_dimension
    TRAIN_SIZE = 1000000
    TEST_SIZE = 10000

    DIM -= 1  # final dimension will be added as offset
    if dsname == SYNTHETIC_NAMES[0]:
        ftype = 'npz'
    else:
        ftype = 'npy'
    train_file = "%s-d-%d-N-%d.%s" % (train, DIM + 1, TRAIN_SIZE, ftype)
    test_file = "%s-d-%d-N-%d.%s" % (test, DIM + 1, TRAIN_SIZE, ftype)
    create_folder_if_not_exist(os.path.dirname(train))
    create_folder_if_not_exist(os.path.dirname(test))
    if os.path.exists(train_file):
        if not os.path.exists(test_file):
            raise ValueError("'%s' exists but '%s' doesn't!" %
                             (train_file, test_file))
        print('Not regenerating data for %s' % dsname)
        return train_file, test_file, ftype
    else:
        print('Generating data with train path %s' % train_file)
    if dsname == SYNTHETIC_NAMES[0]: # synth-cmc
        if DIM == 4:
            # From Table 1(b) in Scott et al (2013)
            theta = np.array([-3, 1.2, -.5, .8,  3.])
            probs = np.array([ 1,  .2,  .3, .5, .01])
        elif DIM == 9:
            theta = np.array([-3, 1.2, -.5, .8, -1.,-.7,  3.,   4.,  3.5,  4.5])
            probs = np.array([ 1,  .2,  .3, .5, .1,  .2, .01, .007, .005, .001])
        else:
            sys.exit("If using --synth-cmc, then dimension must be 5 or 10")
        generate_binary_data(TRAIN_SIZE, probs, theta, train_file, False)
        generate_binary_data(TEST_SIZE, probs, theta, test_file, False)
    elif dsname in SYNTHETIC_NAMES[1:]:  # synth-rev or synth-rev-un
        DIM += 1
        covar = np.eye(DIM)
        means = np.zeros((2, DIM))
        if dsname == SYNTHETIC_NAMES[1]:
            pos_prob = .5
        else:
            pos_prob = .01
        means[0, :DIM//2] = 1.
        means[1, DIM//2:] = 1.
        generate_reverse_mixture(TRAIN_SIZE, pos_prob, means, covar, train_file)
        generate_reverse_mixture(TEST_SIZE, pos_prob, means, covar, test_file)
    else:
        raise ValueError("Unrecognized synthetic dataset '%s'" % dsname)
    return train_file, test_file, ftype


def main():
    args = parse_arguments()

    create_folder_if_not_exist(RESULTS_DIR)
    os.chdir(RESULTS_DIR)
    print('changed working directory to', RESULTS_DIR)

    datasets, test_datasets = construct_datasets(args)
    if not args.skip_run:
        print('running experiment...')
        if args.timing:
            run_timing_experiment(args, datasets, test_datasets)
        else:
            run_experiment(args, datasets, test_datasets)
    if not args.skip_plot:
        print('plotting experiment results...')
        if args.timing:
            make_timing_plots(args, datasets, test_datasets)
        else:
            make_plots(args, datasets, test_datasets)


if __name__ == '__main__':
    main()
