import pandas as pd
from cpsplines.fittings.fit_cpsplines import CPsplines

def cps(x, y, x0):
    data = pd.DataFrame({"x": x, "y": y})
    spl = CPsplines(int_constraints={"x": {1: {"+": 0}}})
    spl.fit(data = data, y_col = "y")
    yhat = spl.predict(data.x)
    y0hat = spl.predict(pd.DataFrame({"x": x0}))
    return yhat, y0hat