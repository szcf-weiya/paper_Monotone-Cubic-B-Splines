from amplpy import AMPL
import numpy as np

ampl = AMPL()
ampl.set_option("solver", "cplex")
ampl.read("ext/1D-regression.increasing.smooth.model")

def nonampl(x, y, lam):
    n = len(x)
    ampl.param["datax"] = x
    ampl.param["datay"] = y
    ampl.param["oneperlambda"] = 1.0 / lam
    ampl.param["cvset"] = np.zeros(n)
    ampl.solve()
    ampl.get_variable("p").get_values().to_pandas()