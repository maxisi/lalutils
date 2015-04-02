import numpy as np
import os
import glob
import sys


class BayesFile(object):
    """
    Hold and manipulate Bayes factors output by lalapps_ppe_nested.
    """
    def __init__(self, model="unknown"):
        self.logB = 0
        self.Zsignal = 0
        self.Znoise = 0
        self.logLmax = 0
        self.model = model
        self.filepath = None

    @classmethod
    def load(cls, filepath, **kwargs):
        new = cls(**kwargs)
        new.filepath = filepath
        data = np.loadtxt(filepath)
        new.logB = data[0]
        new.Zsignal = data[1]
        new.Znoise = data[2]
        new.logLmax = data[3]
        return new


class BayesArray(np.ndarray):
    _defaults = {
        'models': (None, None),
        'origin': None
    }

    def __new__(cls, input_data, **kwargs):
        obj = np.asarray(input_data).view(cls)
        # set metadata using kwargs or default value specified above:
        for key, value in kwargs.iteritems():
            if key not in cls._defaults.keys():
                raise TypeError("BayesArray() got an unexpected argument %r"
                                % key)
            elif key == 'models' and len(value) != 2:
                raise ValueError("invalid 'models' value %r" % value)
        for key, defvalue in cls._defaults.iteritems():
            setattr(obj, key, kwargs.get(key, defvalue))
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set metadata using kwargs or default value specified above:
        for key, value in self._defaults.iteritems():
            setattr(self, key, getattr(obj, key, value))

    @classmethod
    def collect_all(cls, directory, mask="*_B.txt", models=(None, None)):
        if not os.path.isdir(directory):
            raise IOError("invalid source directory %r" % directory)
        pathmask = os.path.join(directory, mask)
        d = []
        for filename in glob.glob(pathmask):
            bf = BayesFile.load(filename)
            d.append(bf.logB)
        return cls(d, origin=directory, models=models)

    @classmethod
    def collect_n(cls, directory, nf, mask="*(N)_B.txt", n0=0,
                  models=(None, None)):
        if not os.path.isdir(directory):
            raise IOError("invalid source directory %r" % directory)
        pathmask = os.path.join(directory, mask)
        d = []
        for n in range(n0, nf):
            pathmask = os.path.join(directory, mask.replace('(N)', str(n)))
            filenames = glob.glob(pathmask)
            if len(filenames) > 1:
                raise IOError("degenerate path mask %r" % pathmask)
            bf = BayesFile.load(filenames[0])
            d.append(bf.logB)
        return cls(d, origin=directory, models=models)

    def export(self, path):
        filename, extension = os.path.splitext(path)
        if '.hdf' in extension:
            raise NotImplementedError("no HDF5 support not yet.")
        else:
            np.savetxt(path, self, fmt="%.6f")


def subtractb(ba1, ba2):
    output = ba1 - ba2
    type1 = ba1.__class__.__name__
    type2 = ba2.__class__.__name__
    if "BayesArray" in type1 and "BayesArray" in type2:
        output.models = (ba1.models[0], ba2.models[0])
    return output