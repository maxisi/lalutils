import numpy as np
import os
import glob
from warnings import warn
from . import basic

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
    def collect_all(cls, directory, mask=None, models=(None, None)):
        mask = mask or "*_B.txt"
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
        d = []
        for n in range(n0, nf):
            path = os.path.join(directory, mask.replace('(N)', str(n)))
            try:
                bf = BayesFile.load(path)
                d.append(bf.logB)
            except IOError:
                warn("Not found: %r" % path)
        return cls(d, origin=directory, models=models)

    def export(self, path):
        filename, extension = os.path.splitext(path)
        if '.hdf' in extension:
            raise NotImplementedError("no HDF5 support not yet.")
        else:
            np.savetxt(path, self, fmt="%.6f")


def subtractb(ba1, ba2):
    """
    :rtype : BayesArray
    :param ba1:
    :param ba2:
    :return:
    """
    output = ba1 - ba2
    type1 = ba1.__class__.__name__
    type2 = ba2.__class__.__name__
    if "BayesArray" in type1 and "BayesArray" in type2:
        output.models = (ba1.models[0], ba2.models[0])
    return output


# CF: lscsoft/src/lalsuite/lalapps/src/pulsar/HeterodyneSearch/pulsarpputils.py
class Prior(object):
    """
    Read and write ppe prior files.
    """

    def __init__(self, model="GR"):
        if not basic.ismodel(model):
            raise AttributeError("invalid signal model %r" % model)
        self.params = {}
        self.model = model
        for param in basic.MODEL_PARAMS[model.upper()]:
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
class PulsarPar(object):
    def __init__(self, **kwargs):
        self.params = {}
        for key, value in kwargs.iteritems():
            self.add(key, value)

    def add(self, key, value):
        key = key.upper()
        self.params[key] = basic.Parameter(key, value)

    def __getitem__(self, item):
        return self.params[item].value

    @classmethod
    def read(cls, path):
        """

        :rtype : PulsarPar
        """
        new = cls()
        paramsadded = 0
        with open(path, 'r') as f:
            for line in f:
                if line[0] not in ['#', ';', '/']:
                    contents = line.split()
                    if not (len(contents) > 4 or len(contents) < 2):
                        if basic.isparam(contents[0]):
                            new.add(contents[0], contents[1])
                            if len(contents) > 2:
                                new.add("%s_ERR" % contents[0], contents[-1])
                        paramsadded += 1
        if paramsadded == 0:
            raise IOError("invalid prior file.")
        else:
            return new

    def write(self, path, params=('RAJ', 'DECJ', 'F0', 'F1', 'PEPOCH')):
        if basic.ismodel(params):
            params = basic.MODEL_PARAMS(params)

        with open(path, 'w') as f:
            for par in params:
                par = par.upper()
                if par in self.params.keys():
                    valuestr = str(self.params[par])
                    f.write("%s %s\n" % (par, str(valuestr)))


class Results(object):
    def __init__(self, bayes=None, injections=None):
        self.bayes = bayes
        self.injections = injections

    @classmethod
    def collect(cls, injpath=None, bpath=None, ninst=None, models=None):
        # check model names
        if models:
            if isinstance(models, basestring):
                if basic.ismodel(models):
                    models = [models]
                else:
                    raise AttributeError('invalid model %r' % models)
            elif 0 < len(models) < 3:
                for m in models:
                    if not (basic.ismodel(m) or m in ['n', 'noise']):
                        raise AttributeError('invalid model %r' % models)
            else:
                raise AttributeError('invalid models %r' % models)

        # load Bayes factors
        # determine filename mask from path
        bdir, bmask = os.path.split(bpath)
        if bmask == '':
            bmask = None
        if ninst:
            if models:
                b = {}
                for m in models:
                    if m not in ['n', 'noise']:
                        b[m] = BayesArray.collect_n(
                            bdir.replace('(M)', m), ninst,
                            mask=bmask.replace('(M)', m), models=(m, 'n'))
                if len(models) == 2:
                    bayes = subtractb(b[models[0]], b[models[1]])
                elif len(models) == 1:
                    bayes = b[models[0]]
                else:
                    raise AttributeError('no valid models in %r' % models)
            else:
                bayes = BayesArray.collect_n(bdir, ninst, mask=bmask)
        else:
            warn('collect_all might return unsorted results.')
            if models:
                b = {}
                for m in models:
                    if m not in ['n', 'noise']:
                        b[m] = BayesArray.collect_all(
                            bpath.replace('(M)', m), mask=bmask,
                            models=(m, 'n'))
                if len(models) == 2:
                    bayes = subtractb(b[models[0]], b[models[1]])
                elif len(models) == 1:
                    bayes = b[models[0]]
                else:
                    raise AttributeError('no valid models in %r' % models)
            else:
                bayes = BayesArray.collect_all(bpath, mask=bmask)

        # load injections
        if injpath:
            if '(N)' not in injpath:
                raise TypeError("injection path name must contain '(N)'")
            else:
                injpars = []
                for ninst in range(len(bayes)):
                    path = injpath.replace('(N)', str(ninst))
                    injpars.append(PulsarPar.read(path))
        return cls(bayes=bayes, injections=injpars)