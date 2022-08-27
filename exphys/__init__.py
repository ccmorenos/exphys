"""Main classes for basic experimental physics."""
import pandas as pd
import numpy as np
from sympy import sympify, lambdify, diff
import matplotlib.pyplot as plt
from pint import UnitRegistry


class DataTable():
    """
    Class that stores the data and the uncertainties.

    Parameters
    ----------
    file: Str, None.
        File of the table in csv format. If None, a new table is used. Default:
        None.

    """

    # Units manage.
    ureg = UnitRegistry()

    # Data Table
    data = pd.DataFrame()
    data_units = dict()

    # Singles.
    singles = dict()
    singles_unc = dict()
    singles_units = dict()

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

    def get_unit(self, asked_unit):
        """
        Return the unit of the measure.

        Parameters
        ----------
        asked_unit: Str.
            Measure or expression of the unit.

        """
        if asked_unit in self.data_units.keys():
            return self.ureg(self.data_units[asked_unit])
        elif asked_unit in self.singles_units.keys():
            return self.ureg(self.singles_units[asked_unit])
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
                if data[i] == 0:
                    continue

                data[i] = np.round(
                    data[i],
                    -np.int(np.floor(np.log10(unc[i])))
                )

        return data

    def set_single(self, name, unit, val, unc):
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

        """
        self.singles[name] = val
        self.singles_unc[name] = unc
        self.singles_units[name] = unit

    def add_column(self, col, unit="", full_val=np.nan, full_unc=np.nan):
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

        """
        if col not in self.data.columns:
            self.data[col] = np.full(len(self.data), full_val)
            self.data[col + "_unc"] = np.full(len(self.data), full_unc)

            self.data_units[col] = unit

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

        col_unc = 0

        for var in vars:
            expr_diff = lambdify(vars, diff(expr, var), "numpy")

            col_unc += (
                self.get_measure(var.name, True) * expr_diff(**cols_args)
            ) ** 2

        unit = f"{col_func(**units_args).units:~P}"

        self.add_column(col, unit)

        self.data[col + "_unc"] = self.round_digits(np.sqrt(col_unc))
        self.data[col] = self.round_unc(
            col_func(**cols_args), np.sqrt(col_unc)
        )

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

        sing_unc = np.linalg.norm(not_nan_uncs / len(not_nan_uncs))

        self.singles_unc[col + "_avg"] = self.round_digits(
            [sing_unc]
        )[0]
        self.singles[col + "_avg"] = self.round_unc(
            [np.mean(self.data[col])], [np.sqrt(sing_unc)]
        )[0]

        self.singles_units[col + "_avg"] = self.data_units[col]

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
                units_args[atom.name] = self.get_unit(atom.name)

                i += 1

        sing_func = lambdify(vars, expr, "numpy")

        sing_unc = 0

        for var in vars:
            expr_diff = lambdify(vars, diff(expr, var), "numpy")
            sing_unc += (
                self.get_measure(var.name, True) * expr_diff(**sings_args)
            ) ** 2

        self.singles_unc[name] = self.round_digits([np.sqrt(sing_unc)])[0]
        self.singles[name] = self.round_unc(
            [sing_func(**sings_args)], [np.sqrt(sing_unc)]
        )[0]

        self.singles_units[name] = f"{sing_func(**units_args).units:~P}"

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

    def plot(self, x_col, y_col, reg=None, show=False):
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

        """
        plt.errorbar(
            self.data[x_col],
            self.data[y_col],
            xerr=self.data[x_col + "_unc"],
            yerr=self.data[y_col + "_unc"], fmt=".", label="data"
        )

        plt.xlabel(
            f"{x_col} [{self.get_unit(self.data_units[x_col]).units:~P}]"
        )
        plt.ylabel(
            f"{y_col} [{self.get_unit(self.data_units[y_col]).units:~P}]"
        )

        if reg:
            (m, delta_m), (b, delta_b), r = self.reg_func[reg](
                self.data[x_col], self.data[y_col]
            )

            m_text = f"({m} +- {delta_m})"
            m_unit = (
                self.get_unit(self.data_units[y_col]) /
                self.get_unit(self.data_units[x_col])
            ).units

            b_text = f"({b} +- {delta_b})"
            b_unit = self.get_unit(self.data_units[y_col])

            reg_data = self.reg_lambda[reg](self.data[x_col], m, b)

            if reg == "lin":
                reg_label = f"{m_text}{m_unit:~P} x + {b_text}{b_unit:~P}"

            elif reg == "log":
                reg_label = f"{b_text}{b_unit:~P} e^({m_text}{m_unit:~P} x)"

            elif reg == "loglog":
                reg_label = f"{b_text}{b_unit:~P} x^({m_text}{m_unit:~P})"

            plt.plot(self.data[x_col], reg_data, label=reg_label)
            plt.plot([], [], " ", label=f"r = {r}")

            plt.legend()

        if show:
            plt.show()

    def print(self):
        """Print all the measures."""
        print("Data table:\n")

        header = "\t|\t".join([
           f"{measure} [{unit}]" for measure, unit in self.data_units.items()
        ])
        print("\t" + header)
        print("-" * (len(header) + 10), end="\n ")

        for ind in self.data.index:

            row = []

            for col_i in range(0, len(self.data.columns), 2):
                val = self.data[self.data.columns[col_i]][ind]
                unc = self.data[self.data.columns[col_i + 1]][ind]

                row.append(f"{val} +- {unc}")

            print("\t|\t".join(row), end="\n ")

        print("-" * (len(header) + 10))

        print("\nSingle measures:\n")

        for key in self.singles:
            val = self.singles[key]
            unc = self.singles_unc[key]

            unit = self.singles_units[key]

            print(f"{key} = ({val} +- {unc}) {unit}")
