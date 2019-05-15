import pandas


class Metric:
    """docstring for Metric."""

    metric_name = None
    label_config = {}
    metric_values = None

    def __init__(self, metric):
        self.metric_name = metric["metric"]["__name__"]
        self.label_config = metric["metric"]
        del self.label_config["__name__"]

        # if it is a single value metric change key name
        if "value" in metric:
            metric["values"] = [metric["value"]]

        self.metric_values = pandas.DataFrame(metric["values"], columns=["ds", "y"]).apply(
            pandas.to_numeric, args=({"errors": "coerce"})
        )
        self.metric_values["ds"] = pandas.to_datetime(self.metric_values["ds"], unit="s")

    def __eq__(self, other):
        return bool(
            (self.metric_name == other.metric_name) and (self.label_config == other.label_config)
        )

    def __str__(self):
        name = "Metric: " + repr(self.metric_name) + "\n"
        labels = "Labels: " + repr(self.label_config) + "\n"
        values = "Values: " + repr(self.metric_values)

        return "{" + "\n" + name + labels + values + "\n" + "}"

    def __add__(self, other):
        if self == other:
            self.metric_values = self.metric_values.append(other.metric_values, ignore_index=True)
            self.metric_values = self.metric_values.dropna()
            self.metric_values = (
                self.metric_values.drop_duplicates("ds")
                .sort_values(by=["ds"])
                .reset_index(drop=True)
            )
            return self
        if self.metric_name != other.metric_name:
            error_string = "Different metric names"
        else:
            error_string = "Different metric labels"
        raise TypeError("Cannot Add different metric types. " + error_string)
