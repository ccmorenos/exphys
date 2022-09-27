"""Main classes for basic experimental physics."""
import pandas as pd
import numpy as np
from sympy import sympify, lambdify, diff
import matplotlib.pyplot as plt
from pint import UnitRegistry
from pint.formatting import formatter


class DataTable():
    """
    Class that stores the data and the uncertainties.

    Parameters
    ----------
    file: Str, None.
        File of the table in csv format. If None, a new table is used. Default:
        None.

    """

    _GRAPH_FORMAT = {
        "as_ratio": False,
        "single_denominator": True,
        "product_fmt": r" ",
        "power_fmt": "{}$^[{}]$",
        "division_fmt": r"{}/{}",
        "parentheses_fmt": r"\left({}\right)",
    }

    _CONSOLE_FORMAT = {
        "as_ratio": True,
        "single_denominator": False,
        "product_fmt": r" ",
        "power_fmt": "{}^[{}]",
        "division_fmt": r"{}/{}",
        "parentheses_fmt": r"\left({}\right)",
    }

    # Units manage.
    ureg = UnitRegistry()

    # Data Table
    data = pd.DataFrame()

    # Singles.
    singles = dict()
    singles_unc = dict()

    # Data properties.
    measure_units = dict()
    measure_sigma = dict()

    def __init__(self, file=None):
        """Construct the data table."""
        if file:
            self.data = pd.read_csv(file)

        # Regression functions.
        self.reg_func = {
            "lin": self.linear_fit,
            "log": self.log_fit,
            "loglog": self.loglog_fit
        }

        self.reg_lambda = {
            "lin": lambda x, m, b: m*x + b,
            "log": lambda x, m, b: np.exp(b+x*m),
            "loglog": lambda x, m, b: np.exp(b) * x**m
        }

    def format_unit(self, unit, FORMAT):
        """
        Format the unit with the format defined.

        Parameters
        ----------
        unit: Str.
            Unit to be formatted.

        FORMAT: Dict.
            Format to use

        """
        abbrv_units = [
            [self.ureg.get_symbol(k), v] for k, v in unit.units._units.items()
        ]

        return formatter(
            abbrv_units, **FORMAT
        ).replace("[", "{").replace("]", "}")

    def console_unit(self, unit):
        """
        Format the unit with the format for console output.

        Parameters
        ----------
        unit: Str.
            Unit to be formatted.

        """
        return self.format_unit(unit, self._CONSOLE_FORMAT)

    def graph_unit(self, unit, par=False):
        """
        Format the unit with the format for graphs.

        Parameters
        ----------
        unit: Str.
            Unit to be formatted.

        par: Bool.
            Flag that specify whether return the unit in parenthesis or not.

        """
        g_unit = self.format_unit(unit, self._GRAPH_FORMAT)

        if par:
            return " (" + g_unit + ")"
        else:
            return g_unit

    def get_unit(self, asked_unit):
        """
        Return the unit of the measure.

        Parameters
        ----------
        asked_unit: Str.
            Measure or expression of the unit.

        """
        if asked_unit in self.measure_units.keys():
            return self.ureg(self.measure_units[asked_unit])
        else:
            return self.ureg(asked_unit)

    def get_measure(self, var, is_unc=False):
        """
        Return the measure from the data o from the singles.

        Parameters
        ----------
        var: Str.
            Name of the measure asked.

        is_unc: Bool.
            Flag that indicates whether the value or the uncertainty of the
            measure is asked.

        Returns
        -------
            Array or number with the measure value or uncertainty.

        """
        if var in self.data.columns:
            return self.data[var + "_unc"] if is_unc else self.data[var]
        elif var in self.singles.keys():
            return self.singles_unc[var] if is_unc else self.singles[var]

    def round_digits(self, data, dig=1):
        """
        Round to the given number of digits.

        Parameters
        ----------
        data: Array.
            Values to be rounded.

        dig: Int.
            Number of digits in the rounding.

        """
        for i in range(len(data)):
            if not np.isnan(data[i]):
                if data[i] == 0:
                    continue

                data[i] = np.round(
                    data[i],
                    -np.int(np.floor(np.log10(np.abs(data[i])))) + dig - 1
                )

        return data

    def round_unc(self, data, unc):
        """
        Round to the given their uncertainties.

        Parameters
        ----------
        data: Array.
            Values to be rounded.

        unc: Array.
            Uncertainties that indicates the number of digits.

        """
        for i in range(len(data)):

            if not np.isnan(data[i]) and not np.isnan(unc[i]):
                if data[i] == 0 or unc[i] == 0:
                    continue

                data[i] = np.round(
                    data[i],
                    -np.int(np.floor(np.log10(unc[i])))
                )

        return data

    def paren_unc(self, unc):
        """
        Int of the uncertainty.

        Parameters
        ----------
        unc: Array.
            Uncertainties that indicates the number of digits.

        """
        if not np.isnan(unc) and unc != 0:
            dig = int(np.floor(np.log10(unc)))

            new_unc = int(unc / 10**dig) if -dig >= 0 else int(unc)

            return new_unc, -dig

        else:
            return (unc, 1)

    def set_single(self, name, unit, val, unc, syst=True):
        """
        Set a single measure.

        Parameters
        ----------
        name: Str.
            Name of the new single measure.

        unit: Str.
            Unit of the measure.

        val: Int, float.
            Value of the measure.

        unc: Int, float.
            Uncertainty of the measure.

        syst: Bool.
            Flag to indicate if the uncertainty is systematic or statistical.
            Default: True.

        """
        self.singles[name] = val
        self.singles_unc[name] = unc
        self.measure_units[name] = unit

        self.measure_sigma[name] = "syst" if syst else "stat"

    def add_column(
        self, col, unit, syst=True, full_val=np.nan, full_unc=np.nan
    ):
        """
        Create a new column and its uncertainty.

        Parameters
        ----------
        col: Str.
            New column name.

        unit: Str.
            Unit of the column.

        full_val: int, Float.
            Default value of the column of values.

        full_unc: int, Float.
            Default value of the column uncertainties.

        syst: Bool.
            Flag to indicate if the uncertainty is systematic or statistical.
            Default: True.

        """
        if col not in self.data.columns:
            self.data[col] = np.full(len(self.data), full_val)
            self.data[col + "_unc"] = np.full(len(self.data), full_unc)

            self.measure_units[col] = unit

            self.measure_sigma[col] = "syst" if syst else "stat"

    def new_value(self, col, val, unc, skip=0):
        """
        Add a new value to the data table.

        Parameters
        ----------
        col: Str.
            New column name.

        val: Int, float.
            Value to ve appended.

        unc: Int, float.
            Uncertainty of the value.

        skip: Int.
            Number of rows to skip.

        """
        # Get index of the last value that is not NaN.
        entry_index = self.data[col].last_valid_index()

        # If the index is less that the length of the data. set the value.
        if (
            (entry_index is None and len(self.data) != 0) or
            (
                entry_index is not None and
                entry_index + 1 + skip < len(self.data)
            )
        ):
            entry_index = 0 if entry_index is None else entry_index + 1 + skip

            self.data[col].iloc[entry_index] = val
            self.data[col + "_unc"].iloc[entry_index] = unc

        # Add a new column if all the column is full of non-NaN values.
        else:
            self.data = pd.concat(
                [self.data, pd.DataFrame({col: [val], col + "_unc": unc})],
                ignore_index=True
            )

    def compute_unc(self, vars, expr, args):
        """
        Compute the uncertainty of a computed quantity.

        Parameters
        ----------
        vars: Array of symbols.
            Variables of the expression.

        expr: Expression.
            Expression from which extract the uncertainty.

        args: Dictionary.
            The variables names and the value it takes in the expression.

        Returns
        -------
            The sum of statical and systematic uncertainty, as well as a flag
            that indicates if there is statistical uncertainties in the
            arguments.

        """
        unc_stat = 0
        unc_syst = 0

        is_stat = False

        max_unc = self.get_measure(vars[0].name, True)

        for var in vars:
            expr_diff = lambdify(vars, diff(expr, var), "numpy")

            taylor = self.get_measure(var.name, True) * expr_diff(**args)

            max_unc = np.max(self.get_measure(var.name, True))

            if self.measure_sigma[var.name] == "stat":
                unc_stat += taylor ** 2
                is_stat = True
            else:
                unc_syst += np.abs(taylor)

        total_unc = unc_syst + np.sqrt(unc_stat)

        for i in range(len(total_unc)):
            if total_unc[i] == 0.0:
                total_unc[i] = max_unc

        return total_unc, is_stat

    def compute_column(self, col, oper):
        """
        Create a new column form the others measures.

        Parameters
        ----------
        col: Str.
            New column name.

        oper: Str.
            Expression in term of the other measures.

        """
        expr = sympify(oper)

        vars = []
        cols_args = dict()
        units_args = dict()

        i = 1

        for atom in expr.atoms():
            if atom.is_symbol:
                vars.append(atom)

                cols_args[atom.name] = self.get_measure(atom.name)
                units_args[atom.name] = i * self.get_unit(atom.name)

                i += 1

        col_func = lambdify(vars, expr, "numpy")

        col_unc, is_stat = self.compute_unc(vars, expr, cols_args)

        unit = f"{col_func(**units_args)}"

        self.add_column(col, unit)

        self.data[col + "_unc"] = self.round_digits(col_unc)
        self.data[col] = self.round_unc(col_func(**cols_args), col_unc)

        self.measure_sigma[col] = "stat" if is_stat else "syst"

    def compute_avg(self, col):
        """
        Create a new single with the average of a column.

        Parameters
        ----------
        col: Str.
            New column name.

        """
        not_nan_uncs = self.data[col + "_unc"][
            self.data[col + "_unc"].notna()
        ]

        sing_unc = (
            (np.max(not_nan_uncs) - np.min(not_nan_uncs)) / 2
            if len(not_nan_uncs) < 10 else
            np.sqrt(
                ((np.mean(not_nan_uncs) - not_nan_uncs) ** 2).sum() /
                len(not_nan_uncs)
            )
        )

        print(sing_unc, len(not_nan_uncs))

        self.singles_unc[col + "_avg"] = self.round_digits(
            [sing_unc]
        )[0]
        self.singles[col + "_avg"] = self.round_unc(
            [np.mean(self.data[col])], [np.sqrt(sing_unc)]
        )[0]

        self.measure_units[col + "_avg"] = self.measure_units[col]

        self.measure_sigma[col + "_avg"] = "stat"

    def compute_single(self, name, oper):
        """
        Create a new single form the others measures.

        Parameters
        ----------
        name: Str.
            New single name.

        oper: Str.
            Expression in term of the other measures.

        """
        expr = sympify(oper)

        vars = []
        sings_args = dict()
        units_args = dict()

        i = 1

        for atom in expr.atoms():
            if atom.is_symbol:
                vars.append(atom)

                sings_args[atom.name] = self.get_measure(atom.name)
                units_args[atom.name] = 1 * self.get_unit(atom.name)

                i += 1

        sing_func = lambdify(vars, expr, "numpy")

        sing_unc, is_stat = self.compute_unc(vars, expr, sings_args)

        self.singles_unc[name] = self.round_digits([sing_unc])[0]
        self.singles[name] = self.round_unc(
            [sing_func(**sings_args)], [sing_unc]
        )[0]

        self.measure_units[name] = f"{sing_func(**units_args)}"

        self.measure_sigma[name] = "stat" if is_stat else "syst"

    def regression_singles(self, x_col, y_col, reg, m_name=None, b_name=None):
        """
        Create a new single form a regression.

        Parameters
        ----------
        x_col: Str.
            Variable to plot in the x axis.

        y_col: Str.
            Variable to plot in the y axis.

        reg: Str, None.
            Run a regression over the data. The options are lin for linear, log
            for semi-logarithmic and loglog for logarithmic regression. If
            None, no regression is run.

        m_name: Str, None.
            Name of the variable of the slope. If None, the variable is no
            saved.

        b_name: Str, None.
            Name of the variable of the intercept. If None, the variable is no
            saved.

        """
        (m, delta_m), (b, delta_b), r = self.reg_func[reg](
            self.data[x_col], self.data[y_col]
        )

        m_unit = (
            self.get_unit(self.measure_units[y_col]) /
            self.get_unit(self.measure_units[x_col])
        ).units

        b_unit = self.get_unit(self.measure_units[y_col]).units

        if m_name:
            self.set_single(m_name, f"{m_unit}", m, delta_m, syst=False)
        if b_name:
            self.set_single(b_name, f"{b_unit}", b, delta_b, syst=False)

    def linear_fit(self, x_data, y_data):
        """
        Fit a linear regression to a set of data.

        Parameters
        ----------
        x_data: Array.
            Independent variable of the regression.

        y_data: Array.
            Dependent variable of the regression.

        Returns
        -------
            Coefficient and intercept of the linear function with errors, as
            well as the pearson coefficient.

        """
        if len(x_data[x_data.notna()]) < len(y_data[y_data.notna()]):
            x_data = x_data[x_data.notna()]
            y_data = y_data[x_data.notna()]
        else:
            x_data = x_data[y_data.notna()]
            y_data = y_data[y_data.notna()]

        # Number of data.
        N = len(x_data)

        # Sums of the data.
        Sx = np.sum(x_data)
        Sy = np.sum(y_data)

        Sxx = np.sum(x_data**2)
        Syy = np.sum(y_data**2)

        Sxy = np.sum(x_data*y_data)

        # Coefficient and intercept.
        m = (Sx*Sy - N*Sxy) / (Sx**2 - N*Sxx)

        b = (Sxy - m*Sxx) / Sx

        # Errors.
        E = sum((y_data - m * x_data - b)**2)

        delta_m = self.round_digits(
            [np.sqrt(E*N / (N-2) / (N*Sxx - Sx**2))]
        )[0]
        delta_b = self.round_digits(
            [np.sqrt(E*Sxx / (N-2) / (N*Sxx - Sx**2))]
        )[0]

        m = self.round_unc([m], [delta_m])[0]
        b = self.round_unc([b], [delta_b])[0]

        # Pearson coefficient.
        r = self.round_digits(
            [(N*Sxy - Sx*Sy) / np.sqrt((N*Sxx - Sx**2)*(N*Syy - Sy**2))], 4
        )[0]

        return (m, delta_m), (b, delta_b), r

    def log_fit(self, x_data, y_data):
        """
        Fit a linear regression to a set of data.

        Parameters
        ----------
        x_data: Array.
            Independent variable of the regression.

        y_data: Array.
            Dependent variable of the regression.

        Returns
        -------
            Coefficient and intercept of the logarithmic function with errors,
            as well as the pearson coefficient.

        """
        return self.linear_fit(x_data, np.log(y_data))

    def loglog_fit(self, x_data, y_data):
        """
        Fit a linear regression to a set of data.

        Parameters
        ----------
        x_data: Array.
            Independent variable of the regression.

        y_data: Array.
            Dependent variable of the regression.

        Returns
        -------
            Coefficient and intercept of the linear function with errors, as
            well as the pearson coefficient.

        """
        return self.linear_fit(np.log(x_data), np.log(y_data))

    def plot(self, x_col, y_col, reg=None, show=False, **kwargs):
        """
        Plot two columns.

        Parameters
        ----------
        x_col: Str.
            Variable to plot in the x axis.

        y_col: Str.
            Variable to plot in the y axis.

        reg: Str, None.
            Run a regression over the data. The options are lin for linear, log
            for semi-logarithmic and loglog for logarithmic regression. If
            None, no regression is run.

        show: Bool.
            Flag that indicates whether to show or not the plot.

        **kwargs:
            x_label: Str.
                Name of the variable in the x axis, if no label is passed, the
                name of the column will be used.

            y_label: Str.
                Name of the variable in the y axis, if no label is passed, the
                name of the column will be used.

        """
        x_label = kwargs.get("x_label", x_col)
        y_label = kwargs.get("y_label", y_col)

        plt.errorbar(
            self.data[x_col],
            self.data[y_col],
            xerr=self.data[x_col + "_unc"],
            yerr=self.data[y_col + "_unc"], fmt=".", label="data"
        )

        x_unit = self.get_unit(self.measure_units[x_col])
        plt.xlabel(("$%s$" % x_label) + self.graph_unit(x_unit, True))

        y_unit = self.get_unit(self.measure_units[y_col])
        plt.ylabel(("$%s$" % y_label) + self.graph_unit(y_unit, True))

        if reg:
            (m, delta_m), (b, delta_b), r = self.reg_func[reg](
                self.data[x_col], self.data[y_col]
            )

            m_text = f"{m} +- {delta_m}"
            m_unit = self.graph_unit(
                self.get_unit(self.measure_units[y_col]) /
                self.get_unit(self.measure_units[x_col])
            )

            b_text = f"{b} +- {delta_b}"
            b_unit = self.graph_unit(self.get_unit(self.measure_units[y_col]))

            reg_data = self.reg_lambda[reg](self.data[x_col], m, b)

            if reg == "lin":
                reg_label = (
                    f"${y_col}$ = {m_text} " + m_unit +
                    f" ${x_col}$ + {b_text} " + b_unit
                )

            elif reg == "log":
                reg_label = (
                    f"${y_col}$ = {b_text} " + b_unit +
                    f" e^({m_text} " + m_unit + " ${x_col}$)"
                )

            elif reg == "loglog":
                reg_label = (
                    f"${y_col}$ = {b_text}" + b_unit +
                    f" ${x_col}$^({m_text} " + m_unit + ")"
                )

            plt.plot(self.data[x_col], reg_data, label=f"{reg_label}")
            plt.plot([], [], " ", label=f"r = {r}")

            plt.legend()

        if show:
            plt.show()

    def hist(self, x_col, n, show=False, **kwargs):
        """
        Plot two columns.

        Parameters
        ----------
        x_col: Str.
            Variable to plot in the x axis.

        n: Int.
            Number of bins in the histogram.

        show: Bool.
            Flag that indicates whether to show or not the plot.

        **kwargs:
            x_label: Str.
                Name of the variable in the x axis, if no label is passed, the
                name of the column will be used.

            y_label: Str.
                Name of the variable in the y axis, if no label is passed, the
                name of the column will be used.

        """
        x_label = kwargs.get("x_label", x_col)
        y_label = kwargs.get("y_label", "Events")

        plt.hist(self.data[x_col], n, label="data")

        x_unit = self.get_unit(self.measure_units[x_col])
        plt.xlabel(("$%s$" % x_label) + self.graph_unit(x_unit, True))

        plt.ylabel(("$%s$ (1/[" % y_label) + self.graph_unit(x_unit) + "])")

        if show:
            plt.show()

    def make_cell(self, content, cell_size=20):
        """Return a string with a cell."""
        space = cell_size - len(content)

        before_spc = " " * int(abs(np.ceil(space/2)))
        after_spc = " " * int(abs(np.floor(space/2)))

        return before_spc + content + after_spc + "|"

    def print(self):
        """Print all the measures."""
        print("Data table:\n")

        header = "|   \\   |"

        for measure, unit in self.measure_units.items():
            if measure in list(self.data.columns):
                col_unit = self.console_unit(self.get_unit(unit))

                header += self.make_cell(f"{measure} [{col_unit}]")

        print("-" * len(header))
        print(header)
        print("-" * len(header))

        for ind in self.data.index:

            row = "|" + self.make_cell(f"{ind}", 7)

            for col_i in range(0, len(self.data.columns), 2):
                val = self.data[self.data.columns[col_i]][ind]
                unc, dig = self.paren_unc(
                    self.data[self.data.columns[col_i + 1]][ind]
                )

                row += self.make_cell(f"%.{max(0, dig)}f({unc})" % val)

            print(row)

        print("-" * len(header))

        print("\nSingle measures:\n")

        for key in self.singles:
            val = self.singles[key]
            unc, dig = self.paren_unc(self.singles_unc[key])

            unit = self.console_unit(self.get_unit(self.measure_units[key]))

            print(f"{key} = %.{dig}f({unc}) {unit}" % val)
