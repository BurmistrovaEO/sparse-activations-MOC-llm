from rich.table import Table
from rich.text import Text
from rich.box import SQUARE_DOUBLE_HEAD

metric_types = ["acc,none", "acc_norm,none", "perplexity"]


def parsed_arguments_table(parsed_arguments):

    table = Table(title="Experiment settings",
                  box=SQUARE_DOUBLE_HEAD, show_lines=True)

    # TODO move to separate class - static fields
    OFF_text = Text()
    OFF_text.append("OFF", style="bold red")

    ON_TEXT_FT = Text()
    num_args_ft = len(parsed_arguments.ft_arg_names)
    for idx, name in enumerate(parsed_arguments.ft_arg_names):
        ON_TEXT_FT.append(f"{name}: ")
        ON_TEXT_FT.append(f"{getattr(parsed_arguments, name)}", style="green")
        if idx < num_args_ft - 1:
            ON_TEXT_FT.append("\n")

    ON_TEXT_SPARSE = Text()
    num_args_sparse = len(parsed_arguments.sparse_arg_names)
    for idx, name in enumerate(parsed_arguments.sparse_arg_names):
        ON_TEXT_SPARSE.append(f"{name}: ")
        ON_TEXT_SPARSE.append(f"{getattr(parsed_arguments, name)}",
                              style="green")
        if idx < num_args_sparse - 1:
            ON_TEXT_SPARSE.append("\n")

    table.add_column("Stage", justify="right",
                     vertical="middle", style="orange3")
    table.add_column("Arguments", justify="left")

    table.add_row(
        "Fine Tuning",
        ON_TEXT_FT if parsed_arguments.lora_finetune
        else OFF_text
    )
    table.add_row("Sparse",
                  ON_TEXT_SPARSE if parsed_arguments.sparsify
                  else OFF_text)
    return table


def results_table(evaluation_results):
    table = Table(title="Evaluation results",
                  box=SQUARE_DOUBLE_HEAD, show_lines=True)

    table.add_column(
        "Metric", justify="right", vertical="middle", style="deep_sky_blue3"
    )

    row_dict = {metric_type: [] for metric_type in metric_types}

    for idx, (name, values) in enumerate(evaluation_results.items()):
        table.add_column(name, justify="center")
        for metric_type in metric_types:
            row_dict[metric_type].append(values.get(metric_type, "-"))

    table.add_column("Avg", justify="left")

    avgs = {}

    for key, val in row_dict.items():
        numberic_vals = [value for value in val if isinstance(value, float)]
        length = len(numberic_vals)
        avgs[key] = (
            round(sum(numberic_vals) / length, 3)
            if length > 0
            else "-")

    for key, val in avgs.items():
        row_dict[key].append(val)

    for row_name, row_values in row_dict.items():
        table.add_row(row_name, *[str(rv) for rv in row_values])
    return table
