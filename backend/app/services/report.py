"""Automatic PDF report generation for trained models (reportlab).

Rebuilds the evaluation on the stored pipeline, regenerates the standard
matplotlib charts and embeds them (plus metric tables) into a multi-page PDF.
"""
from __future__ import annotations

import base64
import io
import json
from datetime import datetime

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.model_selection import train_test_split

from ..models import Dataset, ModelArtifact
from ..services import evaluation as ev
from ..services import preprocessing as pp
from ..services import storage

_BRAND = colors.HexColor("#4f46e5")
_MAX_IMG_WIDTH = 170 * mm


def _embed(data_uri: str):
    """Decode a matplotlib data URI into a (reportlab Image, w_mm, h_mm)."""
    raw = data_uri.split(",", 1)[1]
    buf = io.BytesIO(base64.b64decode(raw))
    reader = ImageReader(buf)
    iw, ih = reader.getSize()
    w = min(_MAX_IMG_WIDTH, iw)
    h = ih * (w / max(iw, 1))
    return Image(buf, width=w, height=h)


def _rebuild_eval(db, m: ModelArtifact) -> dict:
    """Recompute metrics + charts from the persisted pipeline (best-effort)."""
    out = {
        "metrics": (json.loads(m.metrics) if m.metrics else {}).get("metrics", {}),
        "confusion_matrix": (json.loads(m.metrics) if m.metrics else {}).get("confusion_matrix", []),
        "feature_importance": (json.loads(m.metrics) if m.metrics else {}).get("feature_importance"),
        "charts": {},
    }
    class_names = json.loads(m.class_names) if m.class_names else []
    feature_names = json.loads(m.feature_names) if m.feature_names else []

    ds = db.query(Dataset).filter(Dataset.id == m.dataset_id).first()
    if ds is None:
        return out
    try:
        df = pd.read_csv(ds.filepath)
    except Exception:  # noqa: BLE001
        return out

    pipeline = storage.load_pipeline((json.loads(m.pipeline) if m.pipeline else {}).get("path"))
    if not pipeline:
        return out
    target = pipeline.get("target_column")
    encoder = pipeline.get("target_encoder")
    if not target or target not in df.columns or encoder is None:
        return out
    try:
        clean = df[df[target].notna()].copy()
        X = pp.apply_preprocessor(clean, pipeline)
        y = encoder.transform(clean[target].astype(str))
    except Exception:  # noqa: BLE001
        return out
    if y.shape[0] != X.shape[0]:
        return out

    stored = json.loads(m.metrics) if m.metrics else {}
    test_size = stored.get("test_size", 0.2)
    random_state = stored.get("random_state", 42)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    model = storage.load_model_artifact(m.filepath)
    result = ev.evaluate_model(model, X_test, y_test, class_names)
    out["metrics"] = result["metrics"]
    out["confusion_matrix"] = result["confusion_matrix"]
    importance = ev.feature_importance(model, feature_names)
    if importance:
        out["feature_importance"] = importance

    def _try(name: str, fn) -> None:
        try:
            uri = fn()
            if uri:
                out["charts"][name] = uri
        except Exception:  # noqa: BLE001 - charts are best-effort
            pass

    print("[report] entering charts section")
    _try("confusion_matrix", lambda: ev.chart_confusion_matrix(result["confusion_matrix"], class_names))
    if importance:
        _try("feature_importance", lambda: ev.chart_feature_importance(importance))
    _try("class_balance", lambda: ev.chart_class_balance(y_train, class_names))

    y_pred = model.predict(X_test)
    _try("predicted_vs_actual", lambda: ev.chart_predicted_vs_actual(y_test, y_pred, class_names))
    has_proba = hasattr(model, "predict_proba") and len(class_names) >= 2
    if has_proba:
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:  # noqa: BLE001
            y_proba = None
        if y_proba is not None and y_proba.shape[1] == len(class_names):
            _try("roc_curve", lambda: ev.chart_roc_curve(y_test, y_proba, class_names))
            _try("precision_recall", lambda: ev.chart_precision_recall(y_test, y_proba, class_names))
            _try("probability_histogram", lambda: ev.chart_probability_histogram(y_test, y_proba, class_names))
    return out


def _table(data, col_widths, header_bg=_BRAND):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), header_bg),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return t


def build_model_report(m: ModelArtifact, db) -> bytes:
    """Build a multi-page PDF report for a trained model."""
    data = _rebuild_eval(db, m)
    class_names = json.loads(m.class_names) if m.class_names else []
    feature_names = json.loads(m.feature_names) if m.feature_names else []
    metrics = data.get("metrics", {})
    ds = db.query(Dataset).filter(Dataset.id == m.dataset_id).first()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"Model Report — {m.name}",
        author="ModelMind AI",
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Title"], fontSize=22, textColor=_BRAND, spaceAfter=2)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, textColor=_BRAND, spaceBefore=12, spaceAfter=4)
    sub = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#64748b"))
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14)

    story = []

    # Header / cover
    story.append(Paragraph("ModelMind AI", h1))
    story.append(Paragraph("Automatic Model Evaluation Report", sub))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(f"<b>{m.name}</b> &nbsp;·&nbsp; {m.model_type}", body))
    story.append(
        Paragraph(
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} "
            f"&nbsp;·&nbsp; Dataset: {ds.name if ds else '—'} "
            f"({ds.rows if ds else '—'} rows)"
            if ds else f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            sub,
        )
    )
    story.append(Spacer(1, 6 * mm))

    # Executive summary
    story.append(Paragraph("Executive Summary", h2))
    key_rows = [["Metric", "Value"]]
    for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted"):
        v = metrics.get(k)
        if isinstance(v, (int, float)):
            key_rows.append([k.replace("_", " ").title(), f"{v:.4f}" if isinstance(v, float) else str(v)])
    key_rows.append(["Test samples", str(metrics.get("samples", "—"))])
    story.append(_table(key_rows, [60 * mm, 40 * mm]))
    story.append(Spacer(1, 4 * mm))

    # Confusion matrix
    cm = data.get("confusion_matrix", [])
    if cm and class_names:
        story.append(Paragraph("Confusion Matrix", h2))
        rows = [[""] + class_names] + [
            [class_names[i]] + [str(cell) for cell in row] for i, row in enumerate(cm)
        ]
        story.append(_table(rows, [30 * mm] + [26 * mm] * len(class_names)))
        story.append(Spacer(1, 4 * mm))
        if data["charts"].get("confusion_matrix"):
            story.append(_embed(data["charts"]["confusion_matrix"]))

    # Per-class metrics
    per_class = metrics.get("per_class", [])
    if per_class:
        story.append(Paragraph("Per-Class Metrics", h2))
        rows = [["Class", "Precision", "Recall", "F1", "Support"]]
        for pc in per_class:
            rows.append([pc.get("class", ""), f"{pc.get('precision', 0):.4f}", f"{pc.get('recall', 0):.4f}", f"{pc.get('f1', 0):.4f}", str(pc.get("support", ""))])
        story.append(_table(rows, [50 * mm, 30 * mm, 30 * mm, 30 * mm, 25 * mm]))
        story.append(Spacer(1, 4 * mm))

    # Feature importance
    importance = data.get("feature_importance")
    if importance and importance.get("features"):
        story.append(Paragraph("Feature Importance", h2))
        feats = importance["features"][:10]
        vals = importance["importance"][:10]
        rows = [["#", "Feature", "Importance"]] + [
            [str(i + 1), str(f), f"{v:.4f}"] for i, (f, v) in enumerate(zip(feats, vals))
        ]
        story.append(_table(rows, [12 * mm, 70 * mm, 30 * mm]))
        story.append(Spacer(1, 4 * mm))
        if data["charts"].get("feature_importance"):
            story.append(_embed(data["charts"]["feature_importance"]))

    # Performance charts
    chart_sections = [
        ("class_balance", "Class Balance (Training Target)"),
        ("roc_curve", "ROC Curves (One-vs-Rest)"),
        ("precision_recall", "Precision-Recall Curves"),
        ("probability_histogram", "Prediction Confidence"),
        ("predicted_vs_actual", "Predicted vs Actual"),
    ]
    for key, title in chart_sections:
        if data["charts"].get(key):
            story.append(Paragraph(title, h2))
            story.append(_embed(data["charts"][key]))

    # Model params appendix
    params = json.loads(m.params) if m.params else {}
    if params:
        story.append(Paragraph("Hyperparameters", h2))
        rows = [["Parameter", "Value"]] + [[str(k), str(v)] for k, v in params.items()]
        story.append(_table(rows, [60 * mm, 90 * mm]))

    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#94a3b8"))
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"ModelMind AI · page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
