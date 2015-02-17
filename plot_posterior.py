#! /usr/bin/env python

import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
Histograms posterior distribution from file produced by lalapps_nest2pos.
Takes path to posterior file and number of column to plot (0, 1, ...).
"""

parser = argparse.ArgumentParser()
parser.add_argument("column")
parser.add_argument("input_path")
parser.add_argument("output_path")
parser.add_argument("--logscale", action="store_true")
parser.add_argument("-v", "--verbose", action="store_true")

args = parser.parse_args()
verbose = args.verbose
log = args.logscale

# load data
posterior = np.genfromtxt(args.input_path, names=True)
data = posterior[args.column]

plt.figure()
plt.hist(data, 100, log=log, histtype='step', normed=1)
plt.xlabel(args.column)
plt.ylabel("Normed count")
plt.savefig(args.output_path, bbox_inches='tight')
plt.close()

print "Histogram saved: %s" % args.output_path
