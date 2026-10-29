"""PySide6 user interface for facility changes and monthly NO2 data."""

from __future__ import annotations

import html
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from facility_estimates import aggregate_monthly_to_yearly

matplotlib.use("QtAgg")

import matplotlib.dates as mdates  # noqa: E402
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


def display_value(value: object) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value)


class DataFrameModel(QAbstractTableModel):
    def __init__(self, frame: pd.DataFrame, parent: QWidget | None = None):
        super().__init__(parent)
        self.frame = frame.reset_index(drop=True)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.frame.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        column = str(self.frame.columns[index.column()])
        value = self.frame.iat[index.row(), index.column()]
        text = display_value(value)

        if role == Qt.ItemDataRole.DisplayRole:
            return text
        if role == Qt.ItemDataRole.ToolTipRole and text:
            return text
        if role == Qt.ItemDataRole.BackgroundRole:
            if column.startswith("changed_") and text.strip().upper() == "Y":
                return QColor("#fff2a8")
            if column.startswith("Unnamed:"):
                return QColor("#eeeeee")
        if role == Qt.ItemDataRole.TextAlignmentRole and (
            column == "Match" or column.startswith("changed_")
        ):
            return Qt.AlignmentFlag.AlignCenter
        return None

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            name = str(self.frame.columns[section])
            return "" if name.startswith("Unnamed:") else name
        return str(section + 1)


class FacilityFilterProxy(QSortFilterProxyModel):
    def __init__(self, frame: pd.DataFrame, parent: QWidget | None = None):
        super().__init__(parent)
        self.frame = frame
        self.search_text = ""
        self.match_values = {
            str(value).strip() for value in frame.get("Match", pd.Series(dtype=str))
        }
        self.name_columns = [
            column
            for column in frame.columns
            if column.endswith("_infrastructure_name")
            and not column.startswith("changed_")
        ]
        self.changed_columns: set[str] = set()
        self.changed_mode = "All"
        self.setDynamicSortFilter(True)
        self.setSortCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

    def set_search_text(self, text: str) -> None:
        self.search_text = text.strip().casefold()
        self.invalidateFilter()

    def set_match_values(self, values: set[str]) -> None:
        self.match_values = values
        self.invalidateFilter()

    def set_changed_filter(self, columns: set[str], mode: str) -> None:
        self.changed_columns = columns
        self.changed_mode = mode
        self.invalidateFilter()

    def filterAcceptsRow(  # noqa: N802
        self, source_row: int, source_parent: QModelIndex
    ) -> bool:
        row = self.frame.iloc[source_row]
        if str(row.get("Match", "")).strip() not in self.match_values:
            return False

        if self.search_text:
            searchable = " ".join(
                str(row.get(column, ""))
                for column in ["feature_id", *self.name_columns]
            ).casefold()
            if self.search_text not in searchable:
                return False

        if self.changed_columns:
            results = [
                str(row.get(column, "")).strip().upper() == "Y"
                for column in self.changed_columns
            ]
            if self.changed_mode == "All" and not all(results):
                return False
            if self.changed_mode == "Any" and not any(results):
                return False
        return True


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, title: str, parent: QWidget | None = None):
        self.figure = Figure(figsize=(5, 2.7), tight_layout=True)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)
        self.title = title
        self.show_message("Select a facility from the table")

    def show_message(self, message: str) -> None:
        self.axes.clear()
        self.axes.set_title(self.title)
        self.axes.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            wrap=True,
            transform=self.axes.transAxes,
        )
        self.axes.set_axis_off()
        self.draw_idle()

    def plot_mean(self, yearly: pd.DataFrame, monthly: pd.DataFrame) -> None:
        self.axes.clear()
        monthly_plot = monthly.sort_values(["year", "month"]).copy()
        monthly_plot["period"] = pd.to_datetime(
            {
                "year": monthly_plot["year"],
                "month": monthly_plot["month"],
                "day": 1,
            }
        )
        yearly_plot = yearly.dropna(subset=["mean_no2_mol_m2"]).copy()
        yearly_plot["period"] = pd.to_datetime(
            {
                "year": yearly_plot["year"],
                "month": 7,
                "day": 1,
            }
        )
        if monthly_plot["mean_no2_mol_m2"].notna().sum() == 0 and yearly_plot.empty:
            self.show_message("No valid NO₂ observations")
            return
        self.axes.plot(
            monthly_plot["period"],
            monthly_plot["mean_no2_mol_m2"],
            color="#6baed6",
            linewidth=1.2,
            marker="o",
            markersize=2.5,
            label="Monthly mean",
        )
        if not yearly_plot.empty:
            self.axes.plot(
                yearly_plot["period"],
                yearly_plot["mean_no2_mol_m2"],
                color="#084594",
                linestyle="--",
                linewidth=1.3,
                marker="D",
                markersize=5,
                label="Annual / YTD mean",
            )
        self.axes.set_title(self.title)
        self.axes.set_xlabel("Date")
        self.axes.set_ylabel("Tropospheric NO₂ (mol/m²)")
        self.axes.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        self.axes.xaxis.set_major_locator(mdates.YearLocator())
        self.axes.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        self.axes.grid(alpha=0.25)
        self.axes.legend(loc="best")
        self.draw_idle()

    def plot_coverage(self, yearly: pd.DataFrame) -> None:
        self.axes.clear()
        usable = yearly.dropna(subset=["coverage_pct"])
        if usable.empty:
            self.show_message("Coverage is unavailable")
            return
        self.axes.bar(usable["year"].astype(str), usable["coverage_pct"], color="#45a36f")
        self.axes.set_title(self.title)
        self.axes.set_xlabel("Year")
        self.axes.set_ylabel("Days with data (%)")
        self.axes.set_ylim(0, 100)
        self.axes.grid(axis="y", alpha=0.25)
        self.draw_idle()


class DashboardWindow(QMainWindow):
    def __init__(self, project_root: Path):
        super().__init__()
        self.project_root = project_root
        self.mapped = pd.read_csv(
            project_root / "PROCESSED" / "NSTA" / "mapped_surface_points.csv",
            dtype=str,
            keep_default_na=False,
        )
        self.monthly = pd.read_csv(
            project_root / "PROCESSED" / "SENTINEL" / "facility_no2_monthly.csv",
            dtype={"feature_id": str, "Match": str},
        )
        self.yearly = aggregate_monthly_to_yearly(self.monthly)
        self.selected_yearly = pd.DataFrame()

        self.setWindowTitle("Facility changes and Sentinel-5P NO₂")
        self.resize(1500, 950)
        self._build_ui()
        self._connect_signals()
        self._apply_filters()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)

        filters = QGroupBox("Filters")
        filter_layout = QGridLayout(filters)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Feature ID or infrastructure name")
        filter_layout.addWidget(QLabel("Search"), 0, 0)
        filter_layout.addWidget(self.search, 0, 1, 1, 3)

        self.match_checks: dict[str, QCheckBox] = {}
        match_box = QWidget()
        match_layout = QHBoxLayout(match_box)
        match_layout.setContentsMargins(0, 0, 0, 0)
        match_values = sorted(
            {
                str(value).strip()
                for value in self.mapped["Match"]
                if str(value).strip()
            }
        )
        for value in match_values:
            checkbox = QCheckBox(value)
            checkbox.setChecked(True)
            self.match_checks[value] = checkbox
            match_layout.addWidget(checkbox)
        match_layout.addStretch()
        filter_layout.addWidget(QLabel("Match"), 1, 0)
        filter_layout.addWidget(match_box, 1, 1)

        self.changed_mode = QComboBox()
        self.changed_mode.addItems(["All", "Any"])
        filter_layout.addWidget(QLabel("Selected changes"), 1, 2)
        filter_layout.addWidget(self.changed_mode, 1, 3)

        self.changed_list = QListWidget()
        self.changed_list.setMaximumHeight(120)
        for column in [c for c in self.mapped.columns if c.startswith("changed_")]:
            item = QListWidgetItem(column.removeprefix("changed_").replace("_", " "))
            item.setData(Qt.ItemDataRole.UserRole, column)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.changed_list.addItem(item)
        filter_layout.addWidget(QLabel("Changed fields"), 2, 0)
        filter_layout.addWidget(self.changed_list, 2, 1, 1, 2)

        self.clear_filters = QPushButton("Clear filters")
        self.row_count = QLabel()
        filter_layout.addWidget(self.clear_filters, 2, 3)
        filter_layout.addWidget(self.row_count, 3, 0, 1, 4)
        root_layout.addWidget(filters)

        self.source_model = DataFrameModel(self.mapped, self)
        self.proxy_model = FacilityFilterProxy(self.mapped, self)
        self.proxy_model.setSourceModel(self.source_model)
        self.table = QTableView()
        self.table.setModel(self.proxy_model)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.verticalHeader().setVisible(False)
        self.table.setWordWrap(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for index, column in enumerate(self.mapped.columns):
            if column.startswith("Unnamed:"):
                self.table.setColumnWidth(index, 14)
            elif column == "feature_id":
                self.table.setColumnWidth(index, 280)
            elif column == "Match":
                self.table.setColumnWidth(index, 70)
            elif column.startswith("changed_"):
                self.table.setColumnWidth(index, 105)
            else:
                self.table.setColumnWidth(index, 170)

        self._build_column_group_controls(root_layout)

        lower = QSplitter(Qt.Orientation.Horizontal)
        detail_frame = QFrame()
        detail_layout = QVBoxLayout(detail_frame)
        self.details = QLabel("Select a facility from the table")
        self.details.setTextFormat(Qt.TextFormat.RichText)
        self.details.setWordWrap(True)
        detail_layout.addWidget(self.details)

        self.status_message = QLabel("")
        self.status_message.setWordWrap(True)
        detail_layout.addWidget(self.status_message)

        self.stats = QTableWidget(0, 6)
        self.stats.setHorizontalHeaderLabels(
            ["Year", "Mean NO₂", "Coverage", "Valid days", "Raster days", "Observations"]
        )
        self.stats.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.stats.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        detail_layout.addWidget(self.stats)
        self.export_button = QPushButton("Export yearly data…")
        self.export_button.setEnabled(False)
        detail_layout.addWidget(self.export_button)

        chart_frame = QFrame()
        chart_layout = QVBoxLayout(chart_frame)
        self.mean_chart = PlotCanvas("Monthly and annual/YTD tropospheric NO₂")
        self.coverage_chart = PlotCanvas("Annual data coverage")
        self.mean_chart.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.coverage_chart.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        chart_layout.addWidget(self.mean_chart)
        chart_layout.addWidget(self.coverage_chart)

        lower.addWidget(detail_frame)
        lower.addWidget(chart_frame)
        lower.setStretchFactor(0, 1)
        lower.setStretchFactor(1, 2)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.table)
        splitter.addWidget(lower)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root_layout.addWidget(splitter)
        self.setCentralWidget(central)

    def _build_column_group_controls(self, root_layout: QVBoxLayout) -> None:
        controls = QGroupBox("Visible column groups")
        controls_layout = QHBoxLayout(controls)

        self.show_all_columns = QPushButton("Show all")
        self.hide_all_columns = QPushButton("Hide all")
        controls_layout.addWidget(self.show_all_columns)
        controls_layout.addWidget(self.hide_all_columns)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFixedHeight(48)

        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(5)

        self.column_groups: dict[str, list[int]] = {}
        self.column_group_buttons: dict[str, QPushButton] = {}
        columns = list(self.mapped.columns)
        for changed_index, column in enumerate(columns):
            if not column.startswith("changed_"):
                continue
            group_name = column.removeprefix("changed_")
            indices = [changed_index - 2, changed_index - 1, changed_index]
            if changed_index + 1 < len(columns) and str(
                columns[changed_index + 1]
            ).startswith("Unnamed:"):
                indices.append(changed_index + 1)
            self.column_groups[group_name] = indices

            button = QPushButton(group_name.replace("_", " ").title())
            button.setCheckable(True)
            button.setChecked(True)
            button.setToolTip(
                "Show or hide the 2025, 2026 and changed columns for this field"
            )
            button.toggled.connect(
                lambda visible, name=group_name: self._set_column_group_visible(
                    name, visible
                )
            )
            self.column_group_buttons[group_name] = button
            button_layout.addWidget(button)
        button_layout.addStretch()

        scroll.setWidget(button_container)
        controls_layout.addWidget(scroll, 1)
        root_layout.addWidget(controls)

        self.show_all_columns.clicked.connect(lambda: self._set_all_column_groups(True))
        self.hide_all_columns.clicked.connect(lambda: self._set_all_column_groups(False))

    def _set_column_group_visible(self, group_name: str, visible: bool) -> None:
        for column_index in self.column_groups[group_name]:
            self.table.setColumnHidden(column_index, not visible)

    def _set_all_column_groups(self, visible: bool) -> None:
        for group_name, button in self.column_group_buttons.items():
            button.blockSignals(True)
            button.setChecked(visible)
            button.blockSignals(False)
            self._set_column_group_visible(group_name, visible)

    def _connect_signals(self) -> None:
        self.search.textChanged.connect(self._apply_filters)
        for checkbox in self.match_checks.values():
            checkbox.toggled.connect(self._apply_filters)
        self.changed_list.itemChanged.connect(self._apply_filters)
        self.changed_mode.currentTextChanged.connect(self._apply_filters)
        self.clear_filters.clicked.connect(self._reset_filters)
        self.table.selectionModel().selectionChanged.connect(self._selection_changed)
        self.export_button.clicked.connect(self._export_selected)

    def _selected_changed_columns(self) -> set[str]:
        return {
            str(item.data(Qt.ItemDataRole.UserRole))
            for row in range(self.changed_list.count())
            if (item := self.changed_list.item(row)).checkState() == Qt.CheckState.Checked
        }

    def _apply_filters(self, *_args) -> None:
        self.proxy_model.set_search_text(self.search.text())
        self.proxy_model.set_match_values(
            {value for value, checkbox in self.match_checks.items() if checkbox.isChecked()}
        )
        self.proxy_model.set_changed_filter(
            self._selected_changed_columns(), self.changed_mode.currentText()
        )
        self.row_count.setText(
            f"Showing {self.proxy_model.rowCount()} of {len(self.mapped)} facilities"
        )

    def _reset_filters(self) -> None:
        self.search.clear()
        for checkbox in self.match_checks.values():
            checkbox.setChecked(True)
        self.changed_mode.setCurrentText("All")
        self.changed_list.blockSignals(True)
        for row in range(self.changed_list.count()):
            self.changed_list.item(row).setCheckState(Qt.CheckState.Unchecked)
        self.changed_list.blockSignals(False)
        self._apply_filters()

    def _selection_changed(self, *_args) -> None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            return
        source_index = self.proxy_model.mapToSource(selected[0])
        row = self.mapped.iloc[source_index.row()]
        self._show_facility(row)

    def _show_facility(self, mapped_row: pd.Series) -> None:
        feature_id = str(mapped_row["feature_id"])
        facility_row_index = int(mapped_row.name)
        yearly = self.yearly[
            self.yearly["facility_row_index"].astype(int) == facility_row_index
        ].sort_values("year")
        monthly = self.monthly[
            self.monthly["facility_row_index"].astype(int) == facility_row_index
        ].sort_values(["year", "month"])
        self.selected_yearly = yearly.copy()
        if yearly.empty:
            self.details.setText(f"<b>{html.escape(feature_id)}</b>")
            self.status_message.setText("No yearly cache records exist for this facility.")
            self.export_button.setEnabled(False)
            return

        first = yearly.iloc[0]
        name_columns = [
            column
            for column in self.mapped.columns
            if column.endswith("_infrastructure_name")
            and not column.startswith("changed_")
        ]
        source_names = "".join(
            "<b>{} name:</b> {}<br>".format(
                html.escape(column.removesuffix("_infrastructure_name")),
                html.escape(str(mapped_row.get(column, "")) or "—"),
            )
            for column in name_columns
        )
        self.details.setText(
            "<b>Feature ID:</b> {}<br>"
            "{}"
            "<b>Match:</b> {}<br>"
            "<b>Analysis coordinate:</b> {:.6f}, {:.6f} ({} source)<br>"
            "<b>Assigned pixels:</b> {}".format(
                html.escape(feature_id),
                source_names,
                html.escape(str(first["Match"])),
                float(first["analysis_longitude"]),
                float(first["analysis_latitude"]),
                html.escape(str(first["coordinate_source"])),
                int(first["area_pixel_count"]),
            )
        )

        status = str(first["analysis_status"])
        messages = {
            "available": "Raw annual Sentinel-5P tropospheric NO₂ assigned by the legacy nearest-facility method.",
            "outside_no2_bbox": "This facility is outside the downloaded NO₂ bounding box, so no yearly values are available.",
            "no_assigned_pixels": "This facility receives no raster pixels because nearby facilities win every pixel under the legacy nearest-facility partition.",
        }
        self.status_message.setText(messages.get(status, status))

        self.stats.setRowCount(len(yearly))
        for table_row, (_, record) in enumerate(yearly.iterrows()):
            mean = record["mean_no2_mol_m2"]
            coverage = record["coverage_pct"]
            values = [
                str(int(record["year"])),
                "" if pd.isna(mean) else f"{float(mean):.6e}",
                "" if pd.isna(coverage) else f"{float(coverage):.1f}%",
                str(int(record["valid_days"])),
                str(int(record["available_raster_days"])),
                str(int(record["valid_pixel_observations"])),
            ]
            for column, value in enumerate(values):
                self.stats.setItem(table_row, column, QTableWidgetItem(value))

        if status == "available":
            self.mean_chart.plot_mean(yearly, monthly)
            self.coverage_chart.plot_coverage(yearly)
        else:
            self.mean_chart.show_message(messages.get(status, status))
            self.coverage_chart.show_message(messages.get(status, status))
        self.export_button.setEnabled(True)

    def _export_selected(self) -> None:
        if self.selected_yearly.empty:
            return
        feature_id = str(self.selected_yearly.iloc[0]["feature_id"])
        suggested = self.project_root / f"{feature_id}_yearly_no2.csv"
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export yearly NO2 data", str(suggested), "CSV files (*.csv)"
        )
        if not filename:
            return
        try:
            self.selected_yearly.to_csv(filename, index=False)
        except Exception as error:
            QMessageBox.critical(self, "Export failed", str(error))
            return
        self.statusBar().showMessage(f"Exported {filename}", 5000)
