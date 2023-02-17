"""Main classes for basic experimental physics."""
import pandas as pd
import numpy as np
from sympy import sympify, lambdify, diff
import matplotlib.pyplot as plt
from pint import UnitRegistry
from pint.formatting import formatter
from scipy.optimize import curve_fit
import json


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

    # Figure.
    fig = plt.figure()

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
            "lin": lambda x, a_0, a_1: a_0 + a_1*x,
            "log": lambda x, a_0, a_1: a_0 * np.exp(a_1*x),
            "loglog": lambda x, a_0, a_1: a_0 * x**a_1
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
                elif data[i] == np.inf:
                    data[i] = 0
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
            entry_index = (
                0 + skip if entry_index is None else entry_index + 1 + skip
            )

            self.data[col].iloc[entry_index] = val
            self.data[col + "_unc"].iloc[entry_index] = unc

        # Add a new column if all the column is full of non-NaN values.
        else:
            self.data = pd.concat(
                [self.data, pd.DataFrame({col: [val], col + "_unc": unc})],
                ignore_index=True
            )

    def compute_unc(self, vars_vals, expr, args, uncs_pos, uncs_neg):
        """
        Compute the uncertainty of a computed quantity.

        Parameters
        ----------
        vars_vals: Array of symbols.
            Variables of the expression.

        expr: Expression.
            Expression from which extract the uncertainty.

        args: Dictionary.
            The variables names and the value it takes in the expression.

        uncs_pos: Dictionary.
            Variable+ uncertainty.

        uncs_neg: Dictionary.
            Variable - uncertainty.

        Returns
        -------
            The sum of statical and systematic uncertainty, as well as a flag
            that indicates if there is statistical uncertainties in the
            arguments.

        """
        unc_stat = 0
        unc_syst = 0

        is_stat = False

        exp_func = lambdify(vars_vals, expr, "numpy")

        max_unc = self.get_measure(vars_vals[0].name, True)

        for var in vars_vals:
            expr_diff = lambdify(vars_vals, diff(expr, var), "numpy")

            taylor = self.get_measure(var.name, True) * expr_diff(**args)

            max_unc = np.max([
                np.abs(exp_func(**args) - exp_func(**uncs_pos)),
                np.abs(exp_func(**args) - exp_func(**uncs_neg))
            ])

            if self.measure_sigma[var.name] == "stat":
                unc_stat += taylor ** 2
                is_stat = True
            else:
                unc_syst += np.abs(taylor)

        total_unc = unc_syst + np.sqrt(unc_stat)
        try:
            for i in range(len(total_unc)):
                if total_unc[i] == 0.0:
                    total_unc[i] = max_unc

        except TypeError:
            if total_unc == 0.0:
                total_unc = max_unc

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

        vars_vals = []
        cols_args = dict()
        unc_args_pos = dict()
        unc_args_neg = dict()
        units_args = dict()

        i = 2

        for atom in expr.atoms():
            if atom.is_symbol:
                vars_vals.append(atom)

                cols_args[atom.name] = self.get_measure(atom.name)
                unc_args_pos[atom.name] = (
                    self.get_measure(atom.name) +
                    self.get_measure(atom.name, True)
                )
                unc_args_neg[atom.name] = (
                    self.get_measure(atom.name) -
                    self.get_measure(atom.name, True)
                )
                units_args[atom.name] = i * self.get_unit(atom.name)

                i += 1

        col_func = lambdify(vars_vals, expr, "numpy")

        col_unc, is_stat = self.compute_unc(
            vars_vals, expr, cols_args, unc_args_pos, unc_args_neg
        )

        unit = f"{col_func(**units_args)}"

        self.add_column(col, unit)

        self.data[col + "_unc"] = self.round_digits(col_unc)
        # self.data[col] = self.round_unc(col_func(**cols_args), col_unc)
        self.data[col] = col_func(**cols_args)

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

        self.singles_unc[col + "_avg"] = self.round_digits(
            [sing_unc]
        )[0]
        self.singles[col + "_avg"] = self.round_unc(
            [np.mean(self.data[col])], [np.sqrt(sing_unc)]
        )[0]

        self.measure_units[col + "_avg"] = self.measure_units[col]

        self.measure_sigma[col + "_avg"] = "stat"

    def compute_cols_err(self, col_name, *, col1, col2=None, ref=None):
        """
        Compute the error of two columns.

        Parameters
        ----------
        col: Str.
            New column name.

        """
        self.add_column(col_name, "")

        data_col1 = self.data[col1]

        if ref is not None:
            data_ref = self.get_measure(ref)

            self.data[col_name + "_unc"] = 0.001
            self.data[col_name] = abs((data_col1 - data_ref) / data_ref)

        elif col2 is not None:
            data_col2 = self.data[col2]

            self.data[col_name + "_unc"] = 0.001

            self.data[col_name] = abs(
                (data_col1 - data_col2 * 2) / (data_col1 + data_col2)
            )

        self.measure_sigma[col_name] = "syst"

    def compute_singles_err(self, sing_name, *, sing1, sing2=None, ref=None):
        """
        Compute the error of two singles.

        Parameters
        ----------
        sing: Str.
            New column name.

        """
        data_sing1 = self.singles[sing1]

        self.measure_units[sing_name] = ""

        if ref is not None:
            data_ref = self.get_measure(ref)

            self.singles_unc[sing_name] = 0.0001
            self.singles[sing_name] = abs((data_sing1 - data_ref) / data_ref)
            print(data_sing1, data_ref)

        elif sing2 is not None:
            data_sing2 = self.singles[sing2]

            self.singles_unc[sing_name] = 0.0001

            self.singles[sing_name] = abs(
                (data_sing1 - data_sing2 * 2) / (data_sing1 + data_sing2)
            )

        self.measure_sigma[sing_name] = "syst"

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

        vars_vals = []
        sings_args = dict()
        unc_args_pos = dict()
        unc_args_neg = dict()
        units_args = dict()

        i = 2

        for atom in expr.atoms():
            if atom.is_symbol:
                vars_vals.append(atom)

                sings_args[atom.name] = self.get_measure(atom.name)
                unc_args_pos[atom.name] = (
                    self.get_measure(atom.name) +
                    self.get_measure(atom.name, True)
                )
                unc_args_neg[atom.name] = (
                    self.get_measure(atom.name) -
                    self.get_measure(atom.name, True)
                )

                units_args[atom.name] = i * self.get_unit(atom.name)

                i += 1.

        sing_func = lambdify(vars_vals, expr, "numpy")

        sing_unc, is_stat = self.compute_unc(
            vars_vals, expr, sings_args, unc_args_pos, unc_args_neg
        )

        self.singles_unc[name] = self.round_digits([sing_unc])[0]
        self.singles[name] = self.round_unc(
            [sing_func(**sings_args)], [sing_unc]
        )[0]

        self.measure_units[name] = f"{sing_func(**units_args)}"

        self.measure_sigma[name] = "stat" if is_stat else "syst"

    def regression_singles(
        self, x_col, y_col, reg, y_sigma=True,
        a_1_name=None, a_0_name=None,
        a_1_unit="", a_0_unit="", p0=None
    ):
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

        y_sigma: Bool.
            Flag that indicates whether use or not uncertainties in the fit.

        a_1_name: Str, None.
            Name of the variable of the slope. If None, the variable is no
            saved.

        a_0_name: Str, None.
            Name of the variable of the intercept. If None, the variable is no
            saved.

        """
        if reg in self.reg_func.keys():
            (a_1, delta_a_1), (a_0, delta_a_0), r = self.reg_func[reg](
                self.data[x_col], self.data[y_col],
                self.data[y_col+"_unc"] if y_sigma else None, p0=p0
            )
        else:
            (a_1, delta_a_1), (a_0, delta_a_0), r = self.function_fit(
                reg,
                self.data[x_col], self.data[y_col],
                self.data[y_col+"_unc"] if y_sigma else None
            )

        if reg == "lin":
            a_1_unit = self.graph_unit(
                self.get_unit(self.measure_units[y_col]) /
                self.get_unit(self.measure_units[x_col])
            )
            a_0_unit = self.graph_unit(
                self.get_unit(self.measure_units[y_col])
            )

        elif reg == "log":
            a_1_unit = self.graph_unit(
                1 / self.get_unit(self.measure_units[x_col])
            )
            a_0_unit = self.graph_unit(
                self.get_unit(self.measure_units[y_col])
            )

        elif reg == "loglog":
            a_1_unit = self.graph_unit(
                self.get_unit(self.measure_units[x_col]) /
                self.get_unit(self.measure_units[x_col])
            )
            a_0_unit = self.graph_unit(
                self.get_unit(self.measure_units[y_col])
            )

        if a_1_name:
            self.set_single(
                a_1_name, f"{a_1_unit}", a_1, delta_a_1, syst=False
            )
        if a_0_name:
            self.set_single(
                a_0_name, f"{a_0_unit}", a_0, delta_a_0, syst=False
            )

    def function_fit(self, f, x_data, y_data, y_sigma=None, p0=None):
        """
        Fit a function to a set of data.

        Parameters
        ----------
        f: Callable.
            Function of the fit. It must receive three arguments, the
            independent variable and two parameters.

        x_data: Array.
            Independent variable of the regression.

        y_data: Array.
            Dependent variable of the regression.

        y_sigma: Array.
            Uncertainties of the dependent variable.

        Returns
        -------
            Coefficient and intercept of the linear function with errors, as
            well as the Pearson coefficient.

        """
        if len(x_data[x_data.notna()]) < len(y_data[y_data.notna()]):
            x_data = x_data[x_data.notna()]
            y_data = y_data[x_data.notna()]
        else:
            x_data = x_data[y_data.notna()]
            y_data = y_data[y_data.notna()]

        # Fit the linear function.
        coeff, errs = curve_fit(f, x_data, y_data, sigma=y_sigma, p0=p0)

        # Get the resulting parameters and errors.
        print(errs)
        delta_a_0, delta_a_1 = np.sqrt(np.diag(errs))
        delta_a_0, delta_a_1 = self.round_digits([delta_a_0, delta_a_1])

        a_0, a_1 = self.round_unc(coeff, [delta_a_0, delta_a_1])

        # Compute r.
        ss_res = np.sum((y_data - f(x_data, *coeff))**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)

        # Pearson coefficient.
        r = self.round_digits(
            [np.sqrt(1 - ss_res / ss_tot)], 4
        )[0]

        return (a_1, delta_a_1), (a_0, delta_a_0), r

    def linear_fit(self, x_data, y_data, y_sigma=None, p0=None):
        """
        Fit a linear regression to a set of data.

        Parameters
        ----------
        x_data: Array.
            Independent variable of the regression.

        y_data: Array.
            Dependent variable of the regression.

        y_sigma: Array.
            Uncertainties of the dependent variable.

        Returns
        -------
            Coefficient and intercept of the linear function with errors, as
            well as the Pearson coefficient.

        """
        return self.function_fit(
            self.reg_lambda["lin"], x_data, y_data,
            y_sigma=y_sigma, p0=p0
        )

    def log_fit(self, x_data, y_data, y_sigma=None, p0=None):
        """
        Fit a linear regression to a set of data.

        Parameters
        ----------
        x_data: Array.
            Independent variable of the regression.

        y_data: Array.
            Dependent variable of the regression.

        y_sigma: Array.
            Uncertainties of the dependent variable.

        Returns
        -------
            Coefficient and intercept of the logarithmic function with errors,
            as well as the Pearson coefficient.

        """
        return self.function_fit(
            self.reg_lambda["log"], x_data, y_data,
            y_sigma=y_sigma, p0=p0
        )

    def loglog_fit(self, x_data, y_data, y_sigma=None, p0=None):
        """
        Fit a linear regression to a set of data.

        Parameters
        ----------
        x_data: Array.
            Independent variable of the regression.

        y_data: Array.
            Dependent variable of the regression.

        y_sigma: Array.
            Uncertainties of the dependent variable.

        Returns
        -------
            Coefficient and intercept of the linear function with errors, as
            well as the Pearson coefficient.

        """
        return self.function_fit(
            self.reg_lambda["loglog"], x_data, y_data,
            y_sigma=y_sigma, p0=p0
        )

    def save_data(self, label, sep=",", index=False):
        """
        Save the DataTable in a csv file.

        Parameters
        ----------
        label: Str.
            Label for the csv file and the json file.

        sep: Str.
            Separator for the csv file. Default ','.

        index: Bool.
            Flag of wether save or not the index in the csv file.

        """
        self.data.to_csv(f"{label}_data_table.csv", sep=sep, index=index)

        json_dict = dict()

        for key in self.singles:
            val = self.singles[key]
            unc, dig = self.paren_unc(self.singles_unc[key])

            unit = self.console_unit(self.get_unit(self.measure_units[key]))

            json_dict[key] = {
                "val": val,
                "unc": 10**-dig * unc,
                "unit": unit
            }

        with open(f"{label}_singles.json", "w") as json_file:
            json_file.write(json.dumps(json_dict, indent=4))
            json_file.close()

    def plot(
        self, x_col, y_col, *y_cols, reg=None, y_sigma=True,
        show=False, legend=False, grid=False, save_file=None, p0=None,
        **kwargs
    ):
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

        y_sigma: Bool.
            Flag that indicates whether use or not uncertainties in the fit.

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

        y_legend = kwargs.get("y_legend", "data")
        y_legends = kwargs.get("y_legends", ["data"]*len(y_cols))

        ax = plt.axes()

        ax.errorbar(
            self.data[x_col],
            self.data[y_col],
            xerr=self.data[x_col + "_unc"],
            yerr=self.data[y_col + "_unc"], fmt=".", label=y_legend
        )

        for y_col_, y_legend_ in zip(y_cols, y_legends):
            ax.errorbar(
                self.data[x_col],
                self.data[y_col_],
                xerr=self.data[x_col + "_unc"],
                yerr=self.data[y_col_ + "_unc"], fmt=".", label=y_legend_
            )

        x_unit = self.get_unit(self.measure_units[x_col])
        plt.xlabel(("$%s$" % x_label) + self.graph_unit(x_unit, True))

        y_unit = self.get_unit(self.measure_units[y_col])
        plt.ylabel(("$%s$" % y_label) + self.graph_unit(y_unit, True))

        if reg:
            (a_1, delta_a_1), (a_0, delta_a_0), r = self.reg_func[reg](
                self.data[x_col], self.data[y_col],
                self.data[y_col+"_unc"] if y_sigma else None,
                p0=p0
            )

            a_1_text = f"{a_1} +- {delta_a_1}"

            a_0_text = f"{a_0} +- {delta_a_0}"

            x_reg = np.linspace(
                min(self.data[x_col]),
                max(self.data[x_col]), 300
            )

            reg_data = self.reg_lambda[reg](x_reg, a_0, a_1)

            if reg == "lin":
                a_1_unit = self.graph_unit(
                    self.get_unit(self.measure_units[y_col]) /
                    self.get_unit(self.measure_units[x_col])
                )
                a_0_unit = self.graph_unit(
                    self.get_unit(self.measure_units[y_col])
                )

                reg_label = (
                    f"${y_label}$ = {a_1_text} " + a_1_unit +
                    f" ${x_label}$ + {a_0_text} " + a_0_unit
                )

            elif reg == "log":
                a_1_unit = self.graph_unit(
                    1 / self.get_unit(self.measure_units[x_col])
                )
                a_0_unit = self.graph_unit(
                    self.get_unit(self.measure_units[y_col])
                )

                reg_label = (
                    f"${y_label}$ = {a_0_text} " + a_0_unit +
                    f" e^({a_1_text} " + a_1_unit + f" ${x_label}$)"
                )
                plt.yscale("log")

            elif reg == "loglog":
                a_1_unit = self.graph_unit(
                    self.get_unit(self.measure_units[x_col]) /
                    self.get_unit(self.measure_units[x_col])
                )
                a_0_unit = self.graph_unit(
                    self.get_unit(self.measure_units[y_col])
                )
                reg_label = (
                    f"${y_label}$ = {a_0_text}" + a_0_unit +
                    f" ${x_label}$^({a_1_text} " + a_1_unit + ")"
                )
                plt.yscale("log")
                plt.xscale("log")

            reg_label = kwargs.get("reg_label", reg_label)

            ax.plot(x_reg, reg_data, label=f"{reg_label}")
            plt.plot([], [], " ", label=f"r = {r}")

        if legend or (legend and reg is not None):
            plt.legend()

        if grid:
            plt.grid(which="both")

        if save_file:
            plt.savefig(save_file, bbox_inches='tight')

        if show:
            plt.show()

        plt.clf()

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

                row += self.make_cell(f"%.{3}e({unc})" % val)

            print(row)

        print("-" * len(header))

        print("\nSingle measures:\n")

        for key in self.singles:
            val = self.singles[key]

            unc, dig = self.paren_unc(self.singles_unc[key])

            unit = self.console_unit(self.get_unit(self.measure_units[key]))

            print(f"{key} = %.{abs(dig)}f({self.singles_unc[key]}) {unit}" % val)
