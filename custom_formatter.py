import logging
from rich.logging import RichHandler

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.box import SQUARE_DOUBLE_HEAD
from rich.color import ANSI_COLOR_NAMES

logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler(console = Console(width=150))]
    )
log= logging.getLogger(__name__)


def parsed_arguments_table(parsed_arguments):

    table = Table(title="Experiment settings", box=SQUARE_DOUBLE_HEAD, show_lines=True)

    #TODO move to separate class - static fields
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
        ON_TEXT_SPARSE.append(f"{getattr(parsed_arguments, name)}", style="green")
        if idx < num_args_sparse - 1:
            ON_TEXT_SPARSE.append("\n")

    table.add_column("Stage", justify="right", vertical="middle", style="orange3")
    table.add_column("Arguments", justify="left")

    table.add_row("Fine Tuning", ON_TEXT_FT if parsed_arguments.lora_finetune else OFF_text)
    table.add_row("Sparse", ON_TEXT_SPARSE if parsed_arguments.sparsify else OFF_text)
    return table

