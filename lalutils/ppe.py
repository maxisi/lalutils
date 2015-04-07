import numpy as np
import os
import glob
from . import basic

MODEL_PARAMS = {
    'GR': ['C22', 'PHI22', 'COSIOTA', 'PSI'],
    'G4V': ['H0', 'PHI0VECTOR', 'IOTA', 'PSI'],
    'ST': ['H0', 'PHI0TENSOR', 'HSCALARB', 'PHI0SCALAR', 'COSIOTA', 'PSI']
}

FLOAT_PARAMS = ["F", "F0", "F1", "F2", "F3", "F4", "F5", "F6",
                "P", "P0", "P1", "P2", "P3", "P4", "P5", "P6",
                "PEPOCH", "POSEPOCH", "DM", "START", "FINISH", "NTOA",
                "TRES", "TZRMJD", "TZRFRQ", "TZRSITE", "NITS",
                "A1", "XDOT", "E", "ECC", "EDOT", "T0", "PB", "PBDOT", "OM",
                "OMDOT", "EPS1", "EPS2", "EPS1DOT", "EPS2DOT", "TASC",
                "LAMBDA",
                "BETA", "RA_RAD", "DEC_RAD", "GAMMA", "SINI", "M2", "MTOT",
                "FB0", "FB1", "FB2", "ELAT", "ELONG", "PMRA", "PMDEC", "DIST",
                # GW PARAMETERS
                "H0", "COSIOTA", "PSI", "PHI0", "THETA", "I21", "I31", "C22",
                "C21", "PHI22", "PHI21", "SNR", "COSTHETA", "IOTA", "HVECTOR"]

STR_PARAMS = ["FILE", "PSR", "PSRJ", "NAME", "RAJ", "DECJ", "RA", "DEC",
              "EPHEM", "CLK", "BINARY", "UNITS"]

###############################################################################


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


# CF: lscsoft/src/lalsuite/lalapps/src/pulsar/HeterodyneSearch/pulsarpputils.py
class PriorFile(object):
    """
    Read and write ppe prior files.
    """

    def __init__(self, model="GR"):
        self.params = {}
        self.model = model
        for param in MODEL_PARAMS[model.upper()]:
            self.params[param] = None

    def add(self, name, prior, pmin, pmax):
        if name.upper() not in self.params.keys():
            raise KeyError("invalid parameter %r for model %s" %
                           (name, self.model))
        if not isinstance(prior, basestring):
            raise TypeError("prior type must be a string (e.g. 'uniform').")
        self.params[name] = (prior, pmin, pmax)

    def write(self, path):
        with open(path, 'w') as f:
            for k, v in self.params.iteritems():
                if v is not None:
                    f.write("%s %s %r %r\n" % (k, v[0], v[1], v[2]))

    def __getitem__(self, item):
        return self.params[item]

    @classmethod
    def read(cls, path, model="GR"):
        new = cls(model=model)
        paramsadded = 0
        with open(path, 'r') as f:
            for line in f:
                if line[0] in ['#', ';', '/']:
                    break
                contents = line.split(' ')
                if len(contents) != 4:
                    break
                elif contents[0] in new.params.keys():
                    new.add(contents[0], contents[1], float(contents[2]),
                            float(contents[3]))
                    paramsadded += 1
        if paramsadded == 0:
            raise IOError("invalid prior file.")
        else:
            return new


# CF: lscsoft/src/lalsuite/lalapps/src/pulsar/HeterodyneSearch/pulsarpputils.py
class ParFile(object):
    def __init__(self, **kwargs):
        self.params = {}
        for key, value in kwargs.iteritems():
            self.add(key, value)

    def add(self, key, value):
        key = key.upper()
        self.params[key] = basic.format_to_store(key, value)

    def __getitem__(self, item):
        return self.params[item]

    @classmethod
    def read(cls, path):
        new = cls()
        paramsadded = 0
        with open(path, 'r') as f:
            for line in f:
                if line[0] in ['#', ';', '/']:
                    break
                contents = line.split()
                if len(contents) > 4 or len(contents) < 2:
                    break
                else:
                    key = contents[0].upper()
                    if key in FLOAT_PARAMS + STR_PARAMS:
                        new.add(contents[0], contents[1])
                        if len(contents) > 2:
                            new.add("%s_ERR" % contents[0], contents[-1])
                    paramsadded += 1
        if paramsadded == 0:
            raise IOError("invalid prior file.")
        else:
            return new

    def write(self, path):
