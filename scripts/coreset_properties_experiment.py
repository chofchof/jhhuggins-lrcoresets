### Experiment to investigate how the size of coresets grows and distribution
### of sensitivities changes with data size and number of clusters

from __future__ import print_function

import os.path
import sys
import argparse
import cPickle as cpk

import numpy as np
import numpy.random as npr
from scipy.sparse import diags
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('notebook', font_scale=3.5, rc={'lines.linewidth': 3})

from coresets.data import load_data
from coresets.distributions import (log_spherical_gaussian,
                                    log_spherical_gaussian_grad)
from coresets.algorithms import (construct_lr_coreset_with_kmeans,
                                 random_data_subset)
from coresets.utils import (pretty_file_string_from_dict,
                            create_folder_if_not_exist)


class CoresetSizeResults(object):
    def __init__(self, Ns, Rs, Ks, eps, delta):
        self.Ns = Ns
        self.Rs = Rs
        self.Ks = Ks
        self.eps = eps
        self.delta = delta
        self.sizes = None
        self.ess = None

    @staticmethod
    def load(fname):
        with open(fname, 'r') as f:
            r = cpk.load(f)
            assert isinstance(r, CoresetSizeResults)
            return r

    def save(self, fname):
        if fname is None:
            return
        with open(fname, 'w') as f:
            cpk.dump(self, f)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file_path')
    parser.add_argument('-N', '--data-sizes', type=str, default='10000')
    parser.add_argument('-d', '--max-dimension', type=int, default=0)
    parser.add_argument('-R', '--radii', type=str, default='1')
    parser.add_argument('-k', '--num-clusters', type=str, default='4')
    parser.add_argument('-r', '--repetitions', type=int, default=10)
    parser.add_argument('-p', type=float, default=1.0)
    parser.add_argument('--include-offset', action='store_true',
                        help='add dummy feature equal to 1')
    parser.add_argument('--eps', type=float, default=0.24)
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--save-dir', type=str, default='.')
    parser.add_argument('--ftype', type=str, default='svmlight')
    parser.add_argument('--randomize-dims', action='store_true')
    parser.add_argument('--no-legend', action='store_true')
    return parser.parse_args()


def plot_results(data, plot_name, args, x_label, x_values, line_name,
                 line_values, y_label, y_scale=None):
    plt.figure()
    # use reverse range so legend and lines match up
    for i in range(data.shape[1] - 1, -1, -1):
        ys = data[:,i,:]
        plt.errorbar(x_values, np.mean(ys, 1),
                     np.std(ys, 1) / np.sqrt(ys.shape[-1]),
                     hold=True,
                     label='%s = %s' % (line_name, line_values[i]))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if not args.no_legend:
        plt.legend(bbox_to_anchor=(1.3, 1), loc=1, borderaxespad=0.)
        # plt.legend(['%s = %s' % (line_name, v) for v in line_values])
    plt.xscale('log')
    if y_scale is not None:
        plt.yscale(y_scale)
    sns.despine()
    plt.savefig(plot_name + '.pdf', bbox_inches='tight')
    plt.close()


def make_plots(results, args, base_filename):
    n_N, n_R, n_K, _ = results.sizes.shape
    for yscale in ['log', 'linear']:
        if n_R > 1:
            for K_idx in range(n_K):
                plot_name = '%s-k-%d-mean-sens-%s' % (base_filename,
                                                      results.Ks[K_idx],
                                                      yscale)
                plot_results(results.sizes[:,:,K_idx,:], plot_name, args, 'N',
                             results.Ns, 'R', results.Rs, 'Mean Sensitivity',
                             yscale)
                plot_name =  '%s-k-%d-ess-%s' % (base_filename,
                                                 results.Ks[K_idx],
                                                 yscale)
                plot_results(results.ess[:,:,K_idx,:], plot_name, args, 'N',
                             results.Ns, 'R', results.Rs, 'Coreset ESS', yscale)
        if n_K > 1:
            for R_idx in range(n_R):
                plot_name = '%s-R-%s-mean-sens-%s' % (base_filename,
                                                      results.Rs[R_idx],
                                                      yscale)
                plot_results(results.sizes[:,R_idx,:,:], plot_name, args, 'N',
                             results.Ns, 'k', results.Ks, 'Mean Sensitivity',
                             yscale)
                plot_name =  '%s-R-%s-ess-%s' % (base_filename,
                                                 results.Rs[R_idx],
                                                 yscale)
                plot_results(results.ess[:,R_idx,:,:], plot_name, args, 'N',
                             results.Ns, 'k', results.Ks, 'Coreset ESS', yscale)


def make_base_filename(args):
    base = os.path.basename(args.data_file_path).split('.')[0]
    attributes = [('data_sizes', 'Ns'),
                  ('radii', 'Rs'),
                  ('num_clusters', 'Ks'),
                  ('eps', 'eps'),
                  ('delta', 'delta'),
                  ('max_dimension', 'd'),
                  ('p', 'p')]
    attr_dict = { a[1] : getattr(args, a[0]) for a in attributes }
    attr_str = pretty_file_string_from_dict(attr_dict)
    name = '-'.join([base, 'coreset-properties', attr_str])
    return os.path.join(args.save_dir, name)


def calc_ess(nums, p=1):
    assert p >= 1
    if nums.size <= 1:
        return 1.0
    probs = nums / np.sum(nums)
    if p == 1:
        return np.exp(-np.sum(probs * np.log(probs))) / probs.shape[0]
    elif p == np.inf:
        return 1.0 / np.max(probs) / probs.shape[0]
    else:
        return 1.0 / np.sum(probs ** p)**(1.0 / (p - 1.0)) / probs.shape[0]


def run_experiment(args, results_file):
    Ns = map(int, args.data_sizes.split(','))
    Rs = map(float, args.radii.split(','))
    Ks = map(int, args.num_clusters.split(','))
    #output_size_param = (args.eps, args.delta)
    output_size_param = 10

    results = CoresetSizeResults(Ns, Rs, Ks, args.eps, args.delta)

    print('loading data...')
    print('ftype =', args.ftype)
    X, y, _  = load_data(args.data_file_path, args.ftype,
                         include_offset=args.include_offset)
    Z = diags(y).dot(X)
    print('%d total data points of dimension %d' % Z.shape)
    if not args.randomize_dims and args.max_dimension > 0:
        Z = Z[:,:args.max_dimension]

    sizes = -1 * np.ones((len(Ns), len(Rs), len(Ks), args.repetitions))
    ess = -1 * np.ones((len(Ns), len(Rs), len(Ks), args.repetitions))

    for N_idx in range(len(Ns)):
        if Ns[N_idx] == 0:
            Ns[N_idx] = Z.shape[0]
        for R_idx in range(len(Rs)):
            for K_idx in range(len(Ks)):
                print(Ns[N_idx], Rs[R_idx], Ks[K_idx])
                for r in range(args.repetitions):
                    curr_Z, _ = random_data_subset(Z, Ns[N_idx],
                                                   args.max_dimension)
                    _, weights, sensitivities = \
                        construct_lr_coreset_with_kmeans(
                                    curr_Z, Ks[K_idx], R=Rs[R_idx],
                                    output_size_param=output_size_param,
                                    return_sensitivities=True)
                    assert sensitivities.shape[0] == curr_Z.shape[0]
                    sizes[N_idx, R_idx, K_idx, r] = np.mean(
                                                        np.ceil(sensitivities))
                    ess[N_idx, R_idx, K_idx, r] = calc_ess(sensitivities,
                                                           args.p)

    results.sizes = sizes
    results.ess = ess
    results.save(results_file)
    assert np.all(sizes != -1)
    assert np.all(ess != -1)
    return results


def main():
    args = parse_arguments()

    create_folder_if_not_exist(args.save_dir)
    base_filename = make_base_filename(args)
    results_file = base_filename + '.cpk'

    if not os.path.isfile(results_file):
        results = run_experiment(args, results_file)
    else:
        results = CoresetSizeResults.load(results_file)

    make_plots(results, args, base_filename)


if __name__ == '__main__':
    main()
