import numpy as np
import imp

import svm
import oracles

from numpy.testing import assert_almost_equal, assert_array_almost_equal

import py.test

def create_context():
    np.random.seed(100)

    X1 = np.random.multivariate_normal(mean=np.array([-2, -2]), cov=np.eye(2), size=(100))
    X2 = np.random.multivariate_normal(mean=np.array([2, 2]), cov=np.eye(2), size=(100))
    X_simple = np.vstack((X1, X2))
    y1 = np.ones(100)
    y2 = np.ones(100) * (-1)
    y_simple = np.hstack((y1, y2))

    np.random.seed(20)

    X1 = np.random.multivariate_normal(mean=np.array([-1.3, -1.3]), cov=np.eye(2), size=(100))
    X2 = np.random.multivariate_normal(mean=np.array([1.3, 1.3]), cov=np.eye(2), size=(100))
    X_medium = np.vstack((X1, X2))
    y1 = np.ones(100)
    y2 = np.ones(100) * (-1)
    y_medium = np.hstack((y1, y2))

    X1 = np.random.multivariate_normal(mean=np.array([-1, -1]), cov=np.eye(2), size=(100))
    X2 = np.random.multivariate_normal(mean=np.array([1, 1]), cov=np.eye(2), size=(100))
    X_hard = np.vstack((X1, X2))
    y1 = np.ones(100)
    y2 = np.ones(100) * (-1)
    y_hard = np.hstack((y1, y2))
    
    return [(X_simple, y_simple), (X_medium, y_medium), (X_hard, y_hard)]
    
def test_prototypes_primal():
    X, y = create_context()[0]
    clf = svm.SVMSolver(C=1, method='primal')
    clf.fit(X, y, max_iter=1, tolerance=1e-1)
    clf.compute_primal_objective(X, y)
    clf.get_w()
    clf.get_w0()
    
def test_prototypes_dual_linear():
    X, y = create_context()[0]
    clf = svm.SVMSolver(C=1, method='dual', kernel='linear')
    clf.fit(X, y, max_iter=1, tolerance=1e-1)
    clf.compute_dual_objective(X, y)
    clf.get_w()
    clf.get_w0()
    clf = svm.SVMSolver(C=1, method='dual', kernel='rbf')
    clf = svm.SVMSolver(C=1, method='dual', kernel='degree')
    
def test_prototypes_hinge_loss():
    X, y = create_context()[0]
    oracle = oracles.BinaryHinge(C=1)
    oracle.func(np.hstack((np.ones((X.shape[0], 1)),X)), y, np.array([1, 1, 1]))
    oracle.grad(np.hstack((np.ones((X.shape[0], 1)),X)), y, np.array([1, 1, 1]))
    
def test_primal():
    clf = svm.SVMSolver(C=1, method='primal')
    train_data = create_context()
    right_answers = [0.155181424397, 0.366848496543, 0.499541949384]
    
    for i, (X, y) in enumerate(train_data):
        clf.fit(X, y, max_iter=20, tolerance=1e-2)
        assert_almost_equal(right_answers[i], clf.compute_primal_objective(X, y), decimal=3)

def test_dual_linear():
    clf = svm.SVMSolver(C=1, method='dual', kernel='linear')
    train_data = create_context()
    right_answers = [-0.154739908173, -0.366837291787, -0.497199649915]
    
    for i, (X, y) in enumerate(train_data):
        clf.fit(X, y, max_iter=20, tolerance=1e-2)
        assert_almost_equal(right_answers[i], clf.compute_dual_objective(X, y), decimal=3)

def test_dual_rbf():
    clf = svm.SVMSolver(C=1, method='dual', kernel='rbf', gamma=0.5)
    train_data = create_context()
    right_answers = [-0.920604728592, -0.925766134098, -0.931810086624]
    
    for i, (X, y) in enumerate(train_data):
        clf.fit(X, y, max_iter=20, tolerance=1e-2)
        assert_almost_equal(right_answers[i], clf.compute_dual_objective(X, y), decimal=3)

def test_dual_polynomial():
    clf = svm.SVMSolver(C=1, method='dual', kernel='polynomial', degree=3)
    train_data = create_context()
    right_answers = [-0.0418342579619, -0.18144659435, -0.293925228588]
    
    for i, (X, y) in enumerate(train_data):
        clf.fit(X, y, max_iter=20, tolerance=1e-2)
        assert_almost_equal(right_answers[i], clf.compute_dual_objective(X, y), decimal=3)

def test_hinge_loss_func():
    train_data = create_context()
    for i, (X, y) in enumerate(train_data):
        oracle = oracles.BinaryHinge(C=1)
        right_answers = [9.82309619813, 7.59694359953, 6.50025897017]

        w = np.array([2, 1, 2])
        assert_almost_equal(right_answers[i],
                            oracle.func(np.hstack((np.ones((X.shape[0], 1)), X)), y, w),
                            decimal=3)
                            
def test_hinge_loss_grad():
    train_data = create_context()
    for i, (X, y) in enumerate(train_data):
        oracle = oracles.BinaryHinge(C=1)
        right_answers = [np.array([0.005, 3.04246936, 4.13781342]),
                         np.array([0.055, 2.18549106, 3.42822627]), 
                         np.array([0.09, 1.87541578, 3.01742159])]

        w = np.array([2, 1, 2])
        assert_array_almost_equal(right_answers[i],
                                  oracle.grad(np.hstack((np.ones((X.shape[0], 1)), X)), y, w),
                                  decimal=3)