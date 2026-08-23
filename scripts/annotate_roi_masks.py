"""Private napari interface for manual lung-ROI annotation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective1.annotation_workspace import (
    ROLE_POLICIES,
    load_annotation_case,
    load_annotation_worklist,
    resolve_annotation_case,
    save_binary_annotation,
    update_annotation_progress,
    update_focused_qc_review,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually edit private lung ROI masks")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--role", required=True, choices=tuple(ROLE_POLICIES))
    parser.add_argument("--annotator", required=True)
    parser.add_argument(
        "--confirm-locked-test-blind",
        action="store_true",
        help="Required for locked test; confirms predictions will not be consulted",
    )
    parser.add_argument(
        "--qc-audit",
        help="Private annotation-QC CSV; restricts the viewer to flagged cases",
    )
    parser.add_argument(
        "--qc-review-log",
        help="Private focused-QC review CSV written when --qc-audit is used",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotator = args.annotator.strip()
    if not annotator:
        raise ValueError("--annotator must not be empty")
    if args.role == "locked_target_test" and not args.confirm_locked_test_blind:
        raise RuntimeError(
            "Locked target test requires --confirm-locked-test-blind and must be "
            "annotated from scratch without viewing predictions"
        )

    import napari
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QCheckBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    worklist, role_root = load_annotation_worklist(args.workspace, args.role)
    focused_qc = bool(args.qc_audit)
    if focused_qc != bool(args.qc_review_log):
        raise ValueError("--qc-audit and --qc-review-log must be provided together")
    qc_flags_lookup: dict[str, str] = {}
    previously_resolved: set[str] = set()
    if focused_qc:
        import pandas as pd

        qc_path = Path(args.qc_audit).resolve()
        if not qc_path.is_file():
            raise FileNotFoundError(qc_path)
        qc = pd.read_csv(qc_path, keep_default_na=False)
        required_qc = {"candidate_code", "qc_flags", "requires_review"}
        missing_qc = sorted(required_qc - set(qc.columns))
        if missing_qc:
            raise ValueError(f"QC audit is missing columns: {missing_qc}")
        if qc["candidate_code"].duplicated().any():
            raise ValueError("QC audit contains duplicate candidate codes")
        requires_review = qc["requires_review"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
        flagged = qc[requires_review].copy()
        if flagged.empty:
            raise ValueError("QC audit has no flagged cases")
        unknown = sorted(set(flagged["candidate_code"]) - set(worklist["candidate_code"]))
        if unknown:
            raise ValueError("QC audit contains cases outside the role worklist")
        qc_flags_lookup = dict(zip(flagged["candidate_code"], flagged["qc_flags"]))
        worklist = worklist[worklist["candidate_code"].isin(qc_flags_lookup)].copy()
        review_path = Path(args.qc_review_log).resolve()
        if review_path.is_file():
            review = pd.read_csv(review_path, keep_default_na=False)
            required_review = {"candidate_code", "cohort_role", "review_status"}
            missing_review = sorted(required_review - set(review.columns))
            if missing_review:
                raise ValueError(
                    f"Focused-QC review log is missing columns: {missing_review}"
                )
            if review["candidate_code"].duplicated().any():
                raise ValueError("Focused-QC review log contains duplicate cases")
            role_review = review[review["cohort_role"].astype(str) == args.role]
            outside_qc = set(role_review["candidate_code"].astype(str)) - set(
                qc_flags_lookup
            )
            if outside_qc:
                raise ValueError(
                    "Focused-QC review log contains cases outside the flagged set"
                )
            previously_resolved = set(
                role_review.loc[
                    role_review["review_status"].astype(str) == "resolved",
                    "candidate_code",
                ].astype(str)
            )
    cases = [
        resolve_annotation_case(row, role_root, role=args.role)
        for _, row in worklist.iterrows()
    ]
    progress_path = role_root / "annotation_progress.csv"

    class AnnotationController(QWidget):
        def __init__(self, viewer: napari.Viewer) -> None:
            super().__init__()
            self.viewer = viewer
            self.index = 0
            self.image_shape: tuple[int, int] = (0, 0)
            self.labels_layer = None
            self.loaded_mask = None
            self.dirty = False
            self.loading = False
            self.session_reviewed: set[str] = set(previously_resolved)

            self.case_label = QLabel()
            self.source_label = QLabel()
            self.progress_label = QLabel()
            self.qc_label = QLabel()
            self.note = QLineEdit()
            self.note.setPlaceholderText("Optional QC/adjudication note")
            self.needs_review = QCheckBox("Needs second review")

            previous_button = QPushButton("Previous")
            save_button = QPushButton("Save")
            save_next_button = QPushButton("Save && Next Unfinished")
            next_button = QPushButton("Next Unfinished")

            previous_button.clicked.connect(self.previous)
            save_button.clicked.connect(self.save)
            save_next_button.clicked.connect(self.save_and_next)
            next_button.clicked.connect(self.next_unfinished)

            row = QHBoxLayout()
            row.addWidget(previous_button)
            row.addWidget(save_button)
            layout = QVBoxLayout()
            layout.addWidget(self.case_label)
            layout.addWidget(self.source_label)
            layout.addWidget(self.progress_label)
            layout.addWidget(self.qc_label)
            layout.addWidget(self.needs_review)
            layout.addWidget(self.note)
            layout.addLayout(row)
            layout.addWidget(save_next_button)
            layout.addWidget(next_button)
            self.setLayout(layout)

            if focused_qc:
                for position, case in enumerate(cases):
                    if case.candidate_code not in self.session_reviewed:
                        self.index = position
                        break
            else:
                for position, case in enumerate(cases):
                    if not case.output_path.is_file():
                        self.index = position
                        break
            self.load_current()

        def completed_count(self) -> int:
            return sum(case.output_path.is_file() for case in cases)

        def mark_dirty(self, event=None) -> None:
            if not self.loading:
                self.dirty = True
                self.refresh_status()

        def refresh_status(self, source: str | None = None) -> None:
            case = cases[self.index]
            suffix = " — UNSAVED EDITS" if self.dirty else ""
            self.case_label.setText(
                f"{case.candidate_code} | {self.index + 1}/{len(cases)}{suffix}"
            )
            if source is not None:
                self.source_label.setText(f"Mask source: {source}")
            if focused_qc:
                self.progress_label.setText(
                    f"Focused QC reviewed: {len(self.session_reviewed)}/{len(cases)} | "
                    f"Role: {args.role} | Reviewer: {annotator}"
                )
                self.qc_label.setText(
                    f"Automated QC flags: {qc_flags_lookup[case.candidate_code]}"
                )
            else:
                self.progress_label.setText(
                    f"Saved masks: {self.completed_count()}/{len(cases)} | "
                    f"Role: {args.role} | Annotator: {annotator}"
                )
                self.qc_label.setText("")

        def load_current(self) -> None:
            self.loading = True
            case = cases[self.index]
            image, mask, source = load_annotation_case(case)
            self.image_shape = tuple(image.shape)
            self.loaded_mask = mask.astype(np.uint8).copy()
            self.viewer.layers.clear()
            self.viewer.add_image(
                image,
                name=f"CXR — {case.candidate_code}",
                colormap="gray",
            )
            self.labels_layer = self.viewer.add_labels(
                mask.astype(np.uint8),
                name="Lung ROI — edit label 1",
                opacity=0.35,
            )
            self.labels_layer.selected_label = 1
            self.labels_layer.brush_size = max(5, int(min(image.shape) * 0.015))
            self.labels_layer.events.data.connect(self.mark_dirty)
            self.viewer.reset_view()
            self.note.clear()
            self.needs_review.setChecked(args.role == "locked_target_test")
            self.dirty = False
            self.loading = False
            self.refresh_status(source)

        def confirm_navigation(self) -> bool:
            if not self.dirty:
                return True
            answer = QMessageBox.question(
                self,
                "Unsaved mask",
                "Save changes before leaving this case?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if answer == QMessageBox.StandardButton.Save:
                self.save()
                return True
            if answer == QMessageBox.StandardButton.Discard:
                return True
            return False

        def save(self) -> None:
            case = cases[self.index]
            metrics = save_binary_annotation(
                np.asarray(self.labels_layer.data),
                case.output_path,
                expected_shape=self.image_shape,
            )
            update_annotation_progress(
                progress_path,
                candidate_code=case.candidate_code,
                role=args.role,
                annotator=annotator,
                foreground_fraction=float(metrics["foreground_fraction"]),
                needs_review=self.needs_review.isChecked(),
                note=self.note.text(),
            )
            if focused_qc:
                changed = not np.array_equal(
                    np.asarray(self.labels_layer.data, dtype=np.uint8),
                    self.loaded_mask,
                )
                update_focused_qc_review(
                    args.qc_review_log,
                    candidate_code=case.candidate_code,
                    role=args.role,
                    reviewer=annotator,
                    qc_flags=qc_flags_lookup[case.candidate_code],
                    action="corrected" if changed else "approved_as_is",
                    needs_review=self.needs_review.isChecked(),
                    note=self.note.text(),
                )
                self.session_reviewed.add(case.candidate_code)
            self.dirty = False
            self.refresh_status("saved_annotation")

        def move_to(self, index: int) -> None:
            if index == self.index or not self.confirm_navigation():
                return
            self.index = index % len(cases)
            self.load_current()

        def previous(self) -> None:
            self.move_to((self.index - 1) % len(cases))

        def next_unfinished(self) -> None:
            if not self.confirm_navigation():
                return
            for offset in range(1, len(cases) + 1):
                position = (self.index + offset) % len(cases)
                unfinished = (
                    cases[position].candidate_code not in self.session_reviewed
                    if focused_qc
                    else not cases[position].output_path.is_file()
                )
                if unfinished:
                    self.index = position
                    self.load_current()
                    return
            QMessageBox.information(
                self,
                "Role complete",
                (
                    "Every QC-flagged case was reviewed in this session."
                    if focused_qc
                    else "Every case has a saved mask."
                ),
            )

        def save_and_next(self) -> None:
            self.save()
            self.next_unfinished()

    viewer = napari.Viewer(title=f"CXR Lung ROI Annotation — {args.role}")
    controller = AnnotationController(viewer)
    viewer.window.add_dock_widget(
        controller,
        name="ROI Annotation Control",
        area="right",
    )

    @viewer.bind_key("Control-S")
    def save_shortcut(viewer_instance) -> None:
        controller.save()

    @viewer.bind_key("PageDown")
    def next_shortcut(viewer_instance) -> None:
        controller.next_unfinished()

    @viewer.bind_key("PageUp")
    def previous_shortcut(viewer_instance) -> None:
        controller.previous()

    print(
        f"Loaded {len(cases)} private {args.role} cases; "
        f"preannotations_allowed={ROLE_POLICIES[args.role]}; "
        f"focused_qc={focused_qc}; cases_loaded={len(cases)}"
    )
    napari.run()


if __name__ == "__main__":
    main()
