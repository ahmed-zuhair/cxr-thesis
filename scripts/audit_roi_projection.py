"""Image-only napari audit for frontal-view eligibility."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective1.annotation_workspace import (
    PROJECTION_DECISIONS,
    ROLE_POLICIES,
    load_projection_audit_worklist,
    load_projection_image,
    resolve_projection_audit_cases,
    select_flagged_projection_cases,
    update_projection_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit frontal-view eligibility without loading model predictions"
    )
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--role", required=True, choices=tuple(ROLE_POLICIES))
    parser.add_argument("--auditor", required=True)
    parser.add_argument(
        "--review-flagged",
        action="store_true",
        help="Re-review only cases whose existing decision is not eligible_frontal",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    auditor = args.auditor.strip()
    if not auditor:
        raise ValueError("--auditor must not be empty")

    import napari
    import pandas as pd
    from qtpy.QtWidgets import (
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    worklist, role_root = load_projection_audit_worklist(args.workspace, args.role)
    cases = resolve_projection_audit_cases(worklist, role_root, role=args.role)
    audit_path = role_root / "projection_audit.csv"
    if args.review_flagged:
        cases = select_flagged_projection_cases(cases, audit_path)
        if not cases:
            print(f"No flagged projection cases remain for {args.role}.")
            return

    class ProjectionAuditController(QWidget):
        def __init__(self, viewer: napari.Viewer) -> None:
            super().__init__()
            self.viewer = viewer
            self.index = 0
            self.session_reviewed: set[str] = set()
            self.case_label = QLabel()
            self.status_label = QLabel()
            self.decision_label = QLabel()
            self.note = QLineEdit()
            self.note.setPlaceholderText("Optional reason or eligibility note")

            eligible = QPushButton("F — Eligible frontal")
            lateral = QPushButton("L — Ineligible lateral")
            other = QPushButton("O — Other ineligible")
            uncertain = QPushButton("U — Uncertain")
            previous = QPushButton("Previous")
            next_unreviewed = QPushButton(
                "Next flagged" if args.review_flagged else "Next unreviewed"
            )

            eligible.clicked.connect(lambda: self.record("eligible_frontal"))
            lateral.clicked.connect(lambda: self.record("ineligible_lateral"))
            other.clicked.connect(lambda: self.record("ineligible_other"))
            uncertain.clicked.connect(lambda: self.record("uncertain"))
            previous.clicked.connect(self.previous)
            next_unreviewed.clicked.connect(self.next_unreviewed)

            navigation = QHBoxLayout()
            navigation.addWidget(previous)
            navigation.addWidget(next_unreviewed)
            layout = QVBoxLayout()
            layout.addWidget(self.case_label)
            layout.addWidget(self.status_label)
            layout.addWidget(self.decision_label)
            layout.addWidget(self.note)
            layout.addWidget(eligible)
            layout.addWidget(lateral)
            layout.addWidget(other)
            layout.addWidget(uncertain)
            layout.addLayout(navigation)
            self.setLayout(layout)

            if not args.review_flagged:
                reviewed = self.read_audit()
                for position, case in enumerate(cases):
                    if case.candidate_code not in reviewed:
                        self.index = position
                        break
            self.load_current()

        def read_audit(self) -> dict[str, str]:
            if not audit_path.is_file():
                return {}
            frame = pd.read_csv(audit_path, keep_default_na=False)
            return dict(zip(frame["candidate_code"], frame["projection_decision"]))

        def refresh_status(self) -> None:
            case = cases[self.index]
            reviewed = self.read_audit()
            current = reviewed.get(case.candidate_code, "not_reviewed")
            self.case_label.setText(
                f"{case.candidate_code} | {self.index + 1}/{len(cases)} | Role: {args.role}"
            )
            if args.review_flagged:
                status = (
                    f"Focused re-review: {len(self.session_reviewed)}/{len(cases)} | "
                    f"Auditor: {auditor}"
                )
            else:
                status = f"Reviewed: {len(reviewed)}/{len(cases)} | Auditor: {auditor}"
            self.status_label.setText(status)
            self.decision_label.setText(f"Current decision: {current}")

        def load_current(self) -> None:
            case = cases[self.index]
            image = load_projection_image(case)
            self.viewer.layers.clear()
            self.viewer.add_image(
                image,
                name=f"Projection audit — {case.candidate_code}",
                colormap="gray",
            )
            self.viewer.reset_view()
            self.note.clear()
            self.refresh_status()

        def record(self, decision: str) -> None:
            if decision not in PROJECTION_DECISIONS:
                raise ValueError(decision)
            case = cases[self.index]
            update_projection_audit(
                audit_path,
                candidate_code=case.candidate_code,
                role=args.role,
                auditor=auditor,
                decision=decision,
                note=self.note.text(),
            )
            self.session_reviewed.add(case.candidate_code)
            self.next_unreviewed()

        def next_unreviewed(self) -> None:
            reviewed = self.read_audit()
            for offset in range(1, len(cases) + 1):
                position = (self.index + offset) % len(cases)
                already_done = (
                    cases[position].candidate_code in self.session_reviewed
                    if args.review_flagged
                    else cases[position].candidate_code in reviewed
                )
                if not already_done:
                    self.index = position
                    self.load_current()
                    return
            self.refresh_status()
            QMessageBox.information(
                self,
                "Projection audit complete",
                (
                    "Every flagged case was re-reviewed in this session."
                    if args.review_flagged
                    else "Every case in this role has a recorded projection decision."
                ),
            )

        def previous(self) -> None:
            self.index = (self.index - 1) % len(cases)
            self.load_current()

    viewer = napari.Viewer(title=f"Prediction-Blind Projection Audit — {args.role}")
    controller = ProjectionAuditController(viewer)
    viewer.window.add_dock_widget(
        controller,
        name="Projection Eligibility",
        area="right",
    )

    @viewer.bind_key("f")
    def eligible_shortcut(viewer_instance) -> None:
        controller.record("eligible_frontal")

    @viewer.bind_key("l")
    def lateral_shortcut(viewer_instance) -> None:
        controller.record("ineligible_lateral")

    @viewer.bind_key("o")
    def other_shortcut(viewer_instance) -> None:
        controller.record("ineligible_other")

    @viewer.bind_key("u")
    def uncertain_shortcut(viewer_instance) -> None:
        controller.record("uncertain")

    @viewer.bind_key("PageDown")
    def next_shortcut(viewer_instance) -> None:
        controller.next_unreviewed()

    @viewer.bind_key("PageUp")
    def previous_shortcut(viewer_instance) -> None:
        controller.previous()

    print(
        f"Loaded {len(cases)} image-only {args.role} cases; "
        f"review_flagged={args.review_flagged}; "
        "prediction_layers_loaded=0; risk_metrics_loaded=0"
    )
    napari.run()


if __name__ == "__main__":
    main()
