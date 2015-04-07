import numpy as np
from .ppe import FLOAT_PARAMS, MODEL_PARAMS

###############################################################################
# CONSTANTS

SS = 86164.0905  # Seconds in a sidereal day
SIDFREQ = 2 * np.pi / SS  # Sidereal angular frequency of Earth
EARTHRADIUS = 6378.137e3  # Earth radius (m)
C = 299792458.  # Speed of light (m/s)


###############################################################################
# CONVERSIONS (FROM polHTC)

def hmsformat(*args):
    if isinstance(args[0], basestring):
        args = (args,)
    if len(args[0]) == 1:
        # Assume hh:mm:ss format
        if type(args) != str:
            argument = args[0][0]
        else:
            argument = args

        hms = argument.split(':')
        if len(hms) == 3:
            h, m, s = [float(x) for x in hms]
        elif len(hms) == 2:
            m, s = [float(x) for x in hms]
            h = 0.0
        elif len(hms) == 1:
            s = [float(x) for x in hms]
            h = 0.0
            m = 0.0
        else:
            raise AttributeError('hmsformat cannot convert: %r' % argument)

    elif len(args[0]) == 3:
        h, m, s = [float(x) for x in args[0]]

    else:
        raise AttributeError('hmsformat can\'t take %d arguments' % len(args))

    return h, m, s


def hms_rad(*args):
    # Converts hours, minutes, seconds to radians using the sidereal angular
    # frequency of the Earth
    h, m, s = hmsformat(args)
    sec = s + 60. * (m + 60. * h)
    return sec * SIDFREQ


def dms_rad(*args):
    # Converts degrees, minutes, seconds to decimal degrees
    d, m, s = hmsformat(args)
    return np.radians(d + m / 60. + s / (60. ** 2))


def masyr_rads(masyr):
    # Converts milliarcseconds/yr to radians/second
    asyr = masyr * 10 ** -3  # mas/yr to arcseconds/yr
    radyr = asyr * np.pi / 648000.  # as/yr to rad/yr (Wikipedia)
    rads = radyr / SS  # rad/yr to rad/s
    return rads


def mjd_gps(mjd):
    # Converts MJD time to GPS time (taken from LALBarycenter.c line 749)
    tgps = 86400. * (mjd - 44244.) - 51.184
    return tgps


def format_to_print(key, value):
    """
    Prepares PAR argument for printing.
    :param key: argument name.
    :param value: argument value (float or string)
    :return: formatted string
    """
    if value < 0:
        sign = "-"
    else:
        sign = ""

    key = key.upper()
    if key.strip("_ERR") not in FLOAT_PARAMS + STR_PARAMS:
        raise ValueError("invalid key %r" % key)

    if key in FLOAT_PARAMS or isinstance(value, basestring):
        return str(value)
    elif key in ['RA', 'RAJ', 'RA_ERR', 'RAJ_ERR']:
        sec = value / SIDFREQ
        if '_ERR' in key:
            return str(sec)
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        sign = ""
        if s >= 9.9995:
            return "%s%.2d:%.2d:%.5f" % (sign, h, m, s)
        else:
            return "%s%.2d:%.2d:0%.5f" % (sign, h, m, s)
    elif key in ['DEC', 'DECJ', 'DEC_ERR', 'DECJ_ERR']:
        # taken from: lscsoft/src/lalsuite/lalapps/src/pulsar/HeterodyneSearch/
        # pulsarpputils.py
        arc = np.degrees(np.fmod(np.fabs(value), np.pi))
        d = int(arc)
        arc = (arc - d) * 60.0
        m = int(arc)
        s = (arc - m) * 60.0
        if '_ERR' in key:
            return str(s)
        if s >= 9.9995:
            return "%s%.2d:%.2d:%.5f" % (sign, d, m, s)
        else:
            return "%s%.2d:%.2d:0%.5f" % (sign, d, m, s)
    else:
        raise TypeError("cannot format argument %s with value %r"
                        % (key, value))


def format_to_store(key, value):
    """
    Prepare PAR argument for storing.
    :param key: argument name.
    :param value: argument value.
    :return: float or string
    """
    key = key.upper()
    if key.replace("_ERR", '') not in (FLOAT_PARAMS + STR_PARAMS):
        raise ValueError("invalid key %r" % key)
    if key in ['RA_ERR', 'RAJ_ERR']:
        if isinstance(value, basestring):
            floatvalue = hms_rad(0., 0., value)
        elif isinstance(value, float):
            floatvalue = value
        else:
            raise TypeError('invalid %s value type %r' % (key, value))
    elif key in ['DEC_ERR', 'DECJ_ERR']:
        if isinstance(value, basestring):
            floatvalue = dms_rad(0., 0., value)
        elif isinstance(value, float):
            floatvalue = value
        else:
            raise TypeError('invalid %s value type %r' % (key, value))
    else:
        try:
            floatvalue = float(value)
        except ValueError:
            if key in ['RA', 'RAJ']:
                floatvalue = hms_rad(value)
            elif key in ['DEC', 'DECJ']:
                floatvalue = dms_rad(value)
            elif isinstance(value, basestring):
                floatvalue = value
            else:
                raise ValueError('invalid PAR %s value %r.'
                                 % (key, value))
    return floatvalue