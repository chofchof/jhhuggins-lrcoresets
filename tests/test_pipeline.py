from coresets.data import generate_gaussian_synthetic
from coresets.algorithms import (construct_lr_coreset_with_kmeans,
                                 random_data_subset,
                                 full_data)
from coresets.evaluation import (mean_relative_squared_error,
                                 log_likelihood, prediction_error)
from coresets.pipeline import (Experiment, Result,
                               _insert_mean_std, _get_colormap)

from numpy.testing import *
import os, shutil
import numpy as np
import numpy.random as npr
import cPickle as cpk


npr.seed(1)


def test_experiment():
    try:
        # setup synthetic data
        generate_gaussian_synthetic(1000, np.array([0, 0]), np.eye(2),
                                    np.array([5, 5]),
                                    'synth_dataset_1.npy')
        generate_gaussian_synthetic(100, np.array([0, 0]), np.eye(2),
                                    np.array([5, 5]),
                                    'synth_dataset_1_test.npy')
        generate_gaussian_synthetic(1000, np.array([0, 0]), np.eye(2),
                                    np.array([5, -5]),
                                    'synth_dataset_2.npy')
        generate_gaussian_synthetic(100, np.array([0, 0]), np.eye(2),
                                    np.array([5, -5]),
                                    'synth_dataset_2_test.npy')

        # setup experiment
        algs = [('Coreset', construct_lr_coreset_with_kmeans),
                ('Random', random_data_subset),
                ('Full', full_data)]
        prm_dicts = [{'output_size_param': 200, 'K' : 4, 'R' : 5.0}]
        evals = [('MRSE', mean_relative_squared_error),
                 ('TLL', log_likelihood),
                 ('TPE', prediction_error)]
        datasets = [('Synth 1', 'synth_dataset_1.npy', 'npy'),
                    ('Synth 2', 'synth_dataset_2.npy', 'npy')]
        test_datasets = [('Synth 1', 'synth_dataset_1_test.npy', 'npy'),
                         ('Synth 2', 'synth_dataset_2_test.npy', 'npy')]
        expt = Experiment('test_expt', algs, prm_dicts, evals, datasets,
                          test_datasets, target_name='Full')

        # assert experiment has expected structure
        algnames = [a[0] for a in expt.algs]
        assert 'Full' not in algnames
        assert 'Coreset' in algnames
        assert 'Random' in algnames
        assert len(algnames) == 2
        assert expt.target_name == 'Full'
        assert expt.target_alg is not None

        # run the experiment
        # TODO use more reasonable params
        expt.run(1, 100)

        # TODO assert results match expected results after run
        # TODO test output on fake data (make sure it creates the # / names
        # of files expected with expected contents)
    finally:
        #cleanup
        to_clean = ['synth_dataset_1.npy', 'synth_dataset_1_test.npy',
                    'synth_dataset_2.npy', 'synth_dataset_2_test.npy']
        for fn in to_clean:
            try:
                os.remove(fn)
            except OSError:
                pass
        try:
            shutil.rmtree('test_expt')
        except OSError:
            pass


def test_result_save_load():
    try:
        # first create an empty result
        r = Result()
        r.populate('target', 'alg', 'dataset', 4, 50, {'prm1': 3, 'prm2':4})
        try:
            f = open('test_result.pk', 'w')
            cpk.dump(r, f)
        except IOError as e:
            print 'Error: Failure to write results to disk'
            raise
        finally:
            f.close()

        print 'R'
        r.disp()

        # now load without any checking, and then make sure they're equal
        r2 = Result()
        r2.load('test_result.pk')
        print 'R2:'
        r2.disp()
        assert r == r2

        # now load with a parameter mismatch
        r3 = Result()
        r3.populate('target', 'alg', 'dataset', 4, 50, {'prm0': 3, 'prm2':5})
        r3.load('test_result.pk', check=True)
        print 'R3:'
        r3.disp()
        assert r3 != r

        # now load with parameters equal
        r4 = Result()
        r4.populate('target', 'alg', 'dataset', 4, 50, {'prm1': 3, 'prm2':4})
        r4.load('test_result.pk', check=True)
        print 'R4:'
        r4.disp()
        assert r4 == r
    finally:
        # remove the file created during the test
        try:
            os.remove('test_result.pk')
        except OSError:
            pass


def test_insert_mean_std():
    w = []
    li = [1, 2, 3]
    r = None
    fcn = lambda x : li
    _insert_mean_std(w, fcn, r)
    print li
    print w
    assert len(w) == 1
    assert len(w[0]) == 2
    assert w[0][0] == 2
    assert w[0][1] == np.sqrt(2./3.)/np.sqrt(len(li))

    w = []
    li = [1]
    fcn = lambda x : li
    print li
    print w
    _insert_mean_std(w, fcn, r)
    assert len(w) ==1
    assert len(w[0]) == 2
    assert w[0][0] == 1
    assert w[0][1] == 0


def test_get_colormap():
    custom_map = {'Name1' : (1, 1, 1), 'Name2' : (.1, .1, .1)}
    names = ['Name', 'Another Name', 'Name1']
    cmap = _get_colormap(names, 'hls', custom_map)
    print cmap
    assert len(cmap) == 4
    assert cmap['Name1'] == (1, 1, 1)
    assert cmap['Name2'] == (.1, .1, .1)
    assert ('Name' in cmap and 'Another Name' in cmap
            and 'Name1' in cmap and 'Name2' in cmap)
