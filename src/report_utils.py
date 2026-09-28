from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle,
)
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)


# ============================================================
# Helper: select representative rows
# ============================================================

def sample_dataframe(df, max_rows=20):
    """
    Select representative rows distributed across the
    complete DataFrame.

    If df contains fewer than max_rows rows, all rows
    are returned.
    """

    if len(df) <= max_rows:
        return df.copy()

    indices = np.linspace(
        0,
        len(df) - 1,
        max_rows,
        dtype=int,
    )

    return df.iloc[indices].copy()


# ============================================================
# Helper: format values
# ============================================================

def format_value(value):
    """
    Format numerical values for terminal and PDF tables.
    """

    if isinstance(value, (float, np.floating)):

        if abs(value) < 0.001 and value != 0:
            return f"{value:.3e}"

        return f"{value:.6f}"

    return str(value)


# ============================================================
# Terminal output
# ============================================================

def print_experiment_summary(
    title,
    parameters,
    summary_df,
    key_results=None,
    max_rows=20,
):
    """
    Print a compact experiment report in the terminal.
    """

    sampled_df = sample_dataframe(
        summary_df,
        max_rows=max_rows,
    )

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print("\nPARAMETERS")

    for key, value in parameters.items():
        print(
            f"  {key}: {format_value(value)}"
        )

    if key_results:

        print("\nKEY RESULTS")

        for key, value in key_results.items():
            print(
                f"  {key}: {format_value(value)}"
            )

    print(
        f"\nRESULTS "
        f"(showing {len(sampled_df)} "
        f"of {len(summary_df)} rows)"
    )

    print(
        sampled_df.to_string(
            index=False,
        )
    )

    print("=" * 70)
    print()


# ============================================================
# PDF report
# ============================================================

def generate_experiment_report(
    title,
    parameters,
    summary_df,
    results_directory,
    key_results=None,
    figure_paths=None,
    description=None,
    max_rows=20,
    filename_prefix="report",
):
    """
    Generate a timestamped PDF report.

    Parameters
    ----------
    title : str
        Experiment title.

    parameters : dict
        Parameters used for the experiment.

    summary_df : pandas.DataFrame
        Summary results.

    results_directory : str or Path
        Directory where report will be saved.

    key_results : dict, optional
        Important derived results such as critical mass,
        critical frequency, etc.

    figure_paths : list, optional
        Paths to PNG/JPG figures.

    description : str, optional
        Short description of experiment.

    max_rows : int
        Maximum number of table rows included.

    filename_prefix : str
        Prefix for generated PDF filename.

    Returns
    -------
    Path
        Path to generated PDF.
    """

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp = datetime.now()

    timestamp_filename = timestamp.strftime(
        "%Y%m%d_%H%M%S"
    )

    timestamp_display = timestamp.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------

    results_directory = Path(
        results_directory
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_path = (
        results_directory
        / (
            f"{filename_prefix}_"
            f"{timestamp_filename}.pdf"
        )
    )

    # --------------------------------------------------------
    # Document setup
    # --------------------------------------------------------

    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ExperimentTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        spaceAfter=12,
    )

    heading_style = styles["Heading2"]

    normal_style = styles["BodyText"]

    story = []

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    story.append(
        Paragraph(
            title,
            title_style,
        )
    )

    story.append(
        Paragraph(
            f"Generated: {timestamp_display}",
            normal_style,
        )
    )

    story.append(
        Spacer(1, 0.4 * cm)
    )

    # --------------------------------------------------------
    # Description
    # --------------------------------------------------------

    if description:

        story.append(
            Paragraph(
                "Experiment Description",
                heading_style,
            )
        )

        story.append(
            Paragraph(
                description,
                normal_style,
            )
        )

        story.append(
            Spacer(1, 0.4 * cm)
        )

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Experiment Parameters",
            heading_style,
        )
    )

    parameter_data = [
        ["Parameter", "Value"]
    ]

    for key, value in parameters.items():

        parameter_data.append(
            [
                str(key),
                format_value(value),
            ]
        )

    parameter_table = Table(
        parameter_data,
        colWidths=[
            7 * cm,
            9 * cm,
        ],
    )

    parameter_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
            ]
        )
    )

    story.append(parameter_table)

    story.append(
        Spacer(1, 0.5 * cm)
    )

    # --------------------------------------------------------
    # Key results
    # --------------------------------------------------------

    if key_results:

        story.append(
            Paragraph(
                "Key Results",
                heading_style,
            )
        )

        result_data = [
            ["Result", "Value"]
        ]

        for key, value in key_results.items():

            result_data.append(
                [
                    str(key),
                    format_value(value),
                ]
            )

        result_table = Table(
            result_data,
            colWidths=[
                9 * cm,
                7 * cm,
            ],
        )

        result_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey,
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold",
                    ),
                ]
            )
        )

        story.append(result_table)

        story.append(
            Spacer(1, 0.5 * cm)
        )

    # --------------------------------------------------------
    # Results table
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Experiment Results",
            heading_style,
        )
    )

    sampled_df = sample_dataframe(
        summary_df,
        max_rows=max_rows,
    )

    story.append(
        Paragraph(
            (
                f"Showing {len(sampled_df)} "
                f"representative results from "
                f"{len(summary_df)} total runs."
            ),
            normal_style,
        )
    )

    story.append(
        Spacer(1, 0.2 * cm)
    )

    # --------------------------------------------------------
    # Format table headers
    # --------------------------------------------------------

    def format_header(column_name):
        """
        Convert DataFrame column names into readable,
        multi-line PDF table headers.
        """

        header = column_name.replace("_", " ")

        return Paragraph(
            header,
            ParagraphStyle(
                "TableHeader",
                parent=styles["BodyText"],
                fontName="Helvetica-Bold",
                fontSize=6.5,
                leading=8,
                alignment=TA_CENTER,
            )
        )


    table_data = [
        [
            format_header(column)
            for column in sampled_df.columns
        ]
    ]


    for _, row in sampled_df.iterrows():

        table_data.append(
            [
                format_value(value)
                for value in row
            ]
        )

    # Automatically distribute width
    available_width = 18 * cm

    number_columns = len(
        sampled_df.columns
    )

    column_width = (
        available_width
        / number_columns
    )

    result_table = Table(
        table_data,
        colWidths=[
            column_width
        ] * number_columns,
        repeatRows=1,
    )

    result_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.grey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
            ]
        )
    )

    story.append(result_table)

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    if figure_paths:

        story.append(
            PageBreak()
        )

        story.append(
            Paragraph(
                "Figures",
                heading_style,
            )
        )

        for figure_path in figure_paths:

            figure_path = Path(
                figure_path
            )

            if not figure_path.exists():
                print(
                    f"Warning: figure not found: "
                    f"{figure_path}"
                )
                continue

            story.append(
                Spacer(1, 0.3 * cm)
            )

            story.append(
                Paragraph(
                    figure_path.stem.replace(
                        "_",
                        " ",
                    ).title(),
                    styles["Heading3"],
                )
            )

            image = Image(
                str(figure_path),
                width=17 * cm,
                height=10 * cm,
            )

            image.hAlign = "CENTER"

            story.append(image)

            story.append(
                Spacer(1, 0.5 * cm)
            )

    # --------------------------------------------------------
    # Generate PDF
    # --------------------------------------------------------

    document.build(story)

    print(
        f"\nPDF report generated:\n{pdf_path}"
    )

    return pdf_path


# ============================================================
# Complete convenience function
# ============================================================

def document_experiment(
    title,
    parameters,
    summary_df,
    results_directory,
    key_results=None,
    figure_paths=None,
    description=None,
    max_rows=20,
    filename_prefix="report",
):
    """
    Print experiment results to terminal and generate
    timestamped PDF report.
    """

    print_experiment_summary(
        title=title,
        parameters=parameters,
        summary_df=summary_df,
        key_results=key_results,
        max_rows=max_rows,
    )

    return generate_experiment_report(
        title=title,
        parameters=parameters,
        summary_df=summary_df,
        results_directory=results_directory,
        key_results=key_results,
        figure_paths=figure_paths,
        description=description,
        max_rows=max_rows,
        filename_prefix=filename_prefix,
    )
