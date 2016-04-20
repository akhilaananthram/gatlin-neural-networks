import caffe
import os
import tempfile
import unittest

from black_and_white_filter import BlackAndWhiteFilter
from caffe import layers as L
from caffe import params as P
from constants import *
from probability_distribution import ProbabilityDistribution
from soft_max import SoftMax

def net_spec_black_and_white():
    n = caffe.NetSpec()
    n.original = L.DummyData(shape={"dim":[BATCH, 3, SIZE, SIZE]},
                             dummy_data_param={"data_filler": {"type": "xavier"}},
                             ntop=1)
    n.blackandwhite = L.Python(n.original, name="blackandwhite", ntop=1,
                                python_param={"module": "black_and_white_filter",
                                              "layer": "BlackAndWhiteFilter"})

    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write(str(n.to_proto()))

    return f.name

def net_spec_soft_max():
    n = caffe.NetSpec()
    n.relu3 = L.DummyData(shape={"dim":[BATCH, 16, 109, 109]},
                             dummy_data_param={"data_filler": {"type": "xavier"}},
                             ntop=1)
    n.softmax = L.Python(n.relu3, name="softmax", ntop=1,
                                python_param={"module": "soft_max",
                                              "layer": "SoftMax"})

    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write(str(n.to_proto()))

    return f.name

def net_spec_probability_distribution():
    n = caffe.NetSpec()
    n.softmax = L.DummyData(shape={"dim":[BATCH, 16, 109, 109]},
                             dummy_data_param={"data_filler": {"type": "xavier"}},
                             ntop=1)
    n.probabilitydist, n.s_cij = L.Python(n.softmax, name="probabilitydist", ntop=2,
                                          python_param={"module": "probability_distribution",
                                                        "layer": "ProbabilityDistribution"})

    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write(str(n.to_proto()))

    return f.name

class TestBlackAndWhiteLayer(unittest.TestCase):
    def setUp(self):
        """
        Initializes the net
        """
        net_spec_file = net_spec_black_and_white()
        self.net = caffe.Net(net_spec_file, caffe.TRAIN)
        os.remove(net_spec_file)

    def test_forward_sanity(self):
        self.net.forward()

    def test_backwards_sanity(self):
        self.net.forward()
        self.net.backward()

class TestSoftMax(unittest.TestCase):
    def setUp(self):
        """
        Initializes the net
        """
        net_spec_file = net_spec_soft_max()
        self.net = caffe.Net(net_spec_file, caffe.TRAIN)
        os.remove(net_spec_file)

    def test_forward_sanity(self):
        self.net.forward()

    def test_backwards_sanity(self):
        self.net.forward()
        self.net.backward()

class TestProbabilityDistribution(unittest.TestCase):
    def setUp(self):
        """
        Initializes the net
        """
        net_spec_file = net_spec_probability_distribution()
        self.net = caffe.Net(net_spec_file, caffe.TRAIN)
        os.remove(net_spec_file)

    def test_forward_sanity(self):
        self.net.forward()

    def test_backwards_sanity(self):
        self.net.forward()
        self.net.backward()

if __name__ == "__main__":
    unittest.main()
