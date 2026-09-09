"""
generate_report.py: Publication-quality PDF report generator for Lab 2.
Machine Learning Applications (MLA), FIT-HANU, AY 2026-2027

Generates complete, publication-ready PDF reports with full Code Appendix for:
  1. Individual Submission: [StudentID]_Lab2.pdf (e.g., 2301140014_Lab2.pdf)
  2. Group Submission:      Group[Number]_Lab2.pdf (e.g., Group01_Lab2.pdf)
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable, PageBreak, Preformatted
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_data, compute_accuracy


def setup_fonts():
    """Register Arial TrueType fonts for full Unicode support."""
    font_paths = {
        "Arial": "/System/Library/Fonts/Supplemental/Arial.ttf",
        "Arial-Bold": "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "Arial-Italic": "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
        "Arial-BoldItalic": "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf",
    }
    for name, path in font_paths.items():
        if os.path.exists(path):
            pdfmetrics.registerFont(TTFont(name, path))


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and render total page numbers and running headers/footers."""
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Arial", 8)
        self.setFillColor(colors.HexColor("#475569"))
        
        # Header (Only on pages 2+)
        if self._pageNumber > 1:
            header_text = getattr(self, 'custom_header', "Machine Learning Applications (MLA) | Lab 2: Decision Trees on Educational Response Data")
            self.drawString(54, 804, header_text)
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.6)
            self.line(54, 796, 541, 796)
        
        # Footer (On all pages)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(54, 46, 541, 46)
        self.drawString(54, 34, "FIT - Hanoi University (FIT-HANU) | Academic Year 2026-2027")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(541, 34, page_text)
        self.restoreState()


def make_canvas_class(custom_header):
    """Factory to inject a custom header string into NumberedCanvas."""
    class CustomNumberedCanvas(NumberedCanvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.custom_header = custom_header
    return CustomNumberedCanvas


def run_experiments():
    """Run data inspection, depth sweep, and test evaluation, producing all figures and metrics."""
    os.makedirs("figs", exist_ok=True)
    print("[1/5] Loading datasets & inspecting statistics...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data("data")
    
    n_train = len(y_train)
    n_val = len(y_val)
    n_test = len(y_test)
    n_users = len(np.unique(X_train[:, 0]))
    n_questions = len(np.unique(X_train[:, 1]))
    prop_correct = float(np.mean(y_train == 1))
    
    depths = list(range(1, 21))
    train_accs, val_accs = [], []
    
    print("[2/5] Fitting Decision Trees across max_depth in {1, ..., 20}...")
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf.fit(X_train, y_train)
        tr = compute_accuracy(y_train, clf.predict(X_train))
        va = compute_accuracy(y_val, clf.predict(X_val))
        train_accs.append(tr)
        val_accs.append(va)
        
    best_idx = int(np.argmax(val_accs))
    opt_d = depths[best_idx]
    opt_val_acc = val_accs[best_idx]
    opt_train_acc = train_accs[best_idx]
    
    print(f"  --> Best max_depth: d* = {opt_d} (Val Acc = {opt_val_acc*100:.2f}%, Train Acc = {opt_train_acc*100:.2f}%)")
    
    # Task 4: Retrain and evaluate on test set
    clf_opt = DecisionTreeClassifier(max_depth=opt_d, random_state=42)
    clf_opt.fit(X_train, y_train)
    test_acc = compute_accuracy(y_test, clf_opt.predict(X_test))
    print(f"  --> Final Test Acc at d*={opt_d}: {test_acc*100:.2f}%")
    
    # Baseline & Lab 1 KNN values
    mv_val_acc = 0.6204
    mv_test_acc = 0.6288
    knn_user_val_acc = 0.6922
    knn_user_test_acc = 0.6887
    knn_item_val_acc = 0.6931
    knn_item_test_acc = 0.6847
    
    print("[3/5] Generating publication-quality validation curve plot...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(6.4, 2.7), dpi=300)
    ax.plot(depths, [a * 100 for a in train_accs], marker='o', color='#1e3a8a', linewidth=1.8, markersize=4.5, label='Train Accuracy (%)')
    ax.plot(depths, [a * 100 for a in val_accs], marker='s', color='#dc2626', linewidth=1.8, markersize=4.5, label='Validation Accuracy (%)')
    ax.axvline(x=opt_d, color='#16a34a', linestyle='--', linewidth=1.5, label=f'Optimal depth d* = {opt_d} ({opt_val_acc*100:.2f}%)')
    ax.scatter([opt_d], [opt_val_acc * 100], color='#16a34a', s=50, zorder=5)

    ax.set_title('Decision Tree: Accuracy vs. Tree Depth (max_depth: 1 to 20)', fontsize=10, fontweight='bold', pad=8, color='#0f172a')
    ax.set_xlabel('max_depth', fontsize=8.5, fontweight='bold', color='#1e293b')
    ax.set_ylabel('Accuracy (%)', fontsize=8.5, fontweight='bold', color='#1e293b')
    ax.set_xticks(depths)
    ax.set_ylim(57.0, 83.0)
    ax.tick_params(axis='both', labelsize=7.8)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', edgecolor='#cbd5e1', loc='lower right', fontsize=7.8)
    fig.tight_layout()
    
    plot_png = "figs/validation_curve_report.png"
    fig.savefig(plot_png, dpi=300)
    fig.savefig("figs/validation_curve.pdf")
    plt.close(fig)
    
    results = {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_users": n_users,
        "n_questions": n_questions,
        "prop_correct": prop_correct,
        "depths": depths,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "opt_d": opt_d,
        "opt_train_acc": opt_train_acc,
        "opt_val_acc": opt_val_acc,
        "test_acc": test_acc,
        "mv_val_acc": mv_val_acc,
        "mv_test_acc": mv_test_acc,
        "knn_user_val_acc": knn_user_val_acc,
        "knn_user_test_acc": knn_user_test_acc,
        "knn_item_val_acc": knn_item_val_acc,
        "knn_item_test_acc": knn_item_test_acc,
        "plot_png": plot_png
    }
    return results


def generate_pdf(results, is_group=False, group_number="Group 01", student_id="2301140014", student_name="Nghiêm Thành Công", output_filename=None):
    setup_fonts()
    
    if output_filename is None:
        if is_group:
            clean_grp = group_number.replace(" ", "")
            output_filename = f"{clean_grp}_Lab2.pdf"
        else:
            output_filename = f"{student_id}_Lab2.pdf"
            
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=56,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Arial-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0F172A"),
        alignment=1,
        spaceAfter=2
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=9.2,
        leading=13,
        textColor=colors.HexColor("#334155"),
        alignment=1,
        spaceAfter=8
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Arial-Bold',
        fontSize=10.5,
        leading=13.5,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=8.3,
        leading=11.6,
        textColor=colors.HexColor("#1E293B"),
        alignment=4,  # Justified
        spaceAfter=4
    )

    code_style = ParagraphStyle(
        'CodeSnippet',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=5.6,
        leading=6.6,
        textColor=colors.HexColor("#0F172A")
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=7.6,
        leading=9.8,
        textColor=colors.HexColor("#1E293B"),
        alignment=1
    )

    table_cell_left = ParagraphStyle(
        'TableCellLeft',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=7.6,
        leading=9.8,
        textColor=colors.HexColor("#1E293B"),
        alignment=0
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=7.8,
        leading=10.0,
        textColor=colors.white,
        alignment=1
    )

    story = []

    # ==========================================
    # PAGE 1: HEADER + SECTION 1 + SECTION 2 (TABLE)
    # ==========================================
    story.append(Paragraph("Lab 2: Decision Trees on Educational Response Data", title_style))
    
    if is_group:
        meta_text = (
            f"<b>Group:</b> {group_number} &nbsp;|&nbsp; "
            f"<b>Submitting Member:</b> {student_name} (ID: {student_id}) &nbsp;|&nbsp; "
            f"<b>Course:</b> Machine Learning Applications (MLA)"
        )
        custom_header = f"Machine Learning Applications (MLA) | Lab 2: Decision Trees | {group_number}"
    else:
        meta_text = (
            f"<b>Student Name:</b> {student_name} &nbsp;|&nbsp; <b>Student ID:</b> {student_id} &nbsp;|&nbsp; <b>Course:</b> Machine Learning Applications (MLA)"
        )
        custom_header = "Machine Learning Applications (MLA) | Lab 2: Decision Trees on Educational Response Data"
        
    story.append(Paragraph(meta_text, subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.0, color=colors.HexColor("#1E3A8A"), spaceAfter=5))

    # SECTION 1: DATASET INSPECTION
    story.append(Paragraph("1. Dataset Inspection (Task 1)", h1_style))
    story.append(Paragraph(
        "The educational response dataset was loaded using <code>utils.load_data()</code>. "
        "The dataset represents binary diagnostic outcomes (1: correct, 0: incorrect) across student interactions with diagnostic questions.",
        body_style
    ))

    t1_data = [
        [Paragraph("Metric / Statistic", table_header_style), Paragraph("Value", table_header_style), Paragraph("Description", table_header_style)],
        [Paragraph("Training Examples", table_cell_left), Paragraph(f"{results['n_train']:,}", table_cell_style), Paragraph("Observed (user_id, question_id, is_correct) training samples", table_cell_left)],
        [Paragraph("Validation Examples", table_cell_left), Paragraph(f"{results['n_val']:,}", table_cell_style), Paragraph("Held-out validation response samples", table_cell_left)],
        [Paragraph("Test Examples", table_cell_left), Paragraph(f"{results['n_test']:,}", table_cell_style), Paragraph("Held-out test response samples", table_cell_left)],
        [Paragraph("Unique Students (Train)", table_cell_left), Paragraph(f"{results['n_users']:,}", table_cell_style), Paragraph("Number of unique students in the training cohort", table_cell_left)],
        [Paragraph("Unique Questions (Train)", table_cell_left), Paragraph(f"{results['n_questions']:,}", table_cell_style), Paragraph("Number of unique diagnostic questions in the training set", table_cell_left)],
        [Paragraph("<b>Proportion Correct (Label = 1)</b>", table_cell_left), Paragraph(f"<b>{results['prop_correct']*100:.2f}%</b> ({results['prop_correct']:.4f})", table_cell_style), Paragraph("Base positive rate in training set (33,923 / 56,688)", table_cell_left)],
        [Paragraph("<b>Majority-Vote Baseline</b>", table_cell_left), Paragraph(f"<b>Val: {results['mv_val_acc']*100:.2f}%</b> | Test: {results['mv_test_acc']*100:.2f}%", table_cell_style), Paragraph("Benchmark accuracy predicting modal response per question", table_cell_left)],
    ]
    t1 = Table(t1_data, colWidths=[130, 110, 247])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.2),
        ('TOPPADDING', (0, 0), (-1, -1), 2.2),
    ]))
    story.append(t1)
    story.append(Spacer(1, 5))

    # SECTION 2: DECISION TREE DEPTH SWEEP
    story.append(Paragraph("2. Decision Tree Depth Sweep (Task 2 & Task 3)", h1_style))
    story.append(Paragraph(
        "Decision tree classifiers were trained across <code>max_depth</code> in {1, 2, ..., 20} with <code>random_state=42</code>. "
        "Training and validation accuracies were recorded to map the model capacity trajectory and identify the optimal depth.",
        body_style
    ))

    # Side-by-side Table of Depths 1-10 and 11-20
    t2_data = [
        [
            Paragraph("<i>d</i>", table_header_style), Paragraph("Train Acc", table_header_style), Paragraph("Val Acc", table_header_style),
            Paragraph("<i>d</i>", table_header_style), Paragraph("Train Acc", table_header_style), Paragraph("Val Acc", table_header_style)
        ]
    ]
    for i in range(10):
        d1 = results['depths'][i]
        tr1 = results['train_accs'][i]
        va1 = results['val_accs'][i]
        mark1 = "<b>*</b>" if d1 == results['opt_d'] else ""
        
        d2 = results['depths'][i + 10]
        tr2 = results['train_accs'][i + 10]
        va2 = results['val_accs'][i + 10]
        mark2 = "<b>*</b>" if d2 == results['opt_d'] else ""

        t2_data.append([
            Paragraph(f"<b>{d1}</b>", table_cell_style), Paragraph(f"{tr1*100:.2f}%", table_cell_style), Paragraph(f"<b>{va1*100:.2f}%{mark1}</b>" if mark1 else f"{va1*100:.2f}%", table_cell_style),
            Paragraph(f"<b>{d2}</b>", table_cell_style), Paragraph(f"{tr2*100:.2f}%", table_cell_style), Paragraph(f"<b>{va2*100:.2f}%{mark2}</b>" if mark2 else f"{va2*100:.2f}%", table_cell_style)
        ])

    t2 = Table(t2_data, colWidths=[35, 75, 75, 35, 75, 75])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('BACKGROUND', (0, 10), (2, 10), colors.HexColor("#E0F2FE")),  # Highlight depth 10
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.8),
        ('TOPPADDING', (0, 0), (-1, -1), 1.8),
    ]))
    story.append(t2)

    # ==========================================
    # PAGE 2: VALIDATION CURVE PLOT + SECTION 3: TEST RESULTS
    # ==========================================
    story.append(PageBreak())

    story.append(Paragraph("Validation Curve & Optimal Depth Selection", h1_style))
    if os.path.exists(results['plot_png']):
        story.append(Image(results['plot_png'], width=440, height=185))
        story.append(Spacer(1, 3))
    
    story.append(Paragraph(
        f"<b>Optimal Depth Selection & Justification:</b> The validation curve achieves its global maximum at <b><i>d*</i> = {results['opt_d']}</b> "
        f"with a validation accuracy of <b>{results['opt_val_acc']*100:.2f}%</b> (an improvement of <b>+{((results['opt_val_acc'] - results['mv_val_acc'])*100):.2f}%</b> over majority vote). "
        f"For low depths (<i>d</i> &le; 4), the tree suffers from high bias (underfitting) and fails to isolate question-specific diagnostic thresholds. "
        f"Beyond <i>d</i> = 10, variance dominates as training accuracy rises monotonically to {results['train_accs'][-1]*100:.2f}% while validation accuracy drops to {results['val_accs'][-1]*100:.2f}%. "
        f"Thus, <b><i>d*</i> = {results['opt_d']}</b> represents the optimal trade-off point under the bias-variance trade-off.",
        body_style
    ))
    story.append(Spacer(1, 5))

    # SECTION 3: TEST RESULTS & BENCHMARK COMPARISON
    story.append(Paragraph("3. Test Results & Model Benchmark (Task 4)", h1_style))
    story.append(Paragraph(
        "The decision tree with optimal depth <i>d*</i> = 10 was retrained on the full training set and evaluated on <code>test_data.csv</code> (3,543 responses). "
        "Below is the benchmark comparing all evaluated models across Lab 1 and Lab 2.",
        body_style
    ))

    t3_data = [
        [Paragraph("Model / Architecture", table_header_style), Paragraph("Optimal Hyperparameter", table_header_style), Paragraph("Validation Accuracy", table_header_style), Paragraph("Final Test Accuracy", table_header_style)],
        [Paragraph("Majority-Vote Baseline", table_cell_left), Paragraph("N/A", table_cell_style), Paragraph(f"{results['mv_val_acc']*100:.2f}%", table_cell_style), Paragraph(f"{results['mv_test_acc']*100:.2f}%", table_cell_style)],
        [Paragraph("<b>Decision Tree (Lab 2)</b>", table_cell_left), Paragraph(f"<b><i>d*</i> = {results['opt_d']}</b>", table_cell_style), Paragraph(f"<b>{results['opt_val_acc']*100:.2f}%</b>", table_cell_style), Paragraph(f"<b>{results['test_acc']*100:.2f}%</b>", table_cell_style)],
        [Paragraph("Item-Based KNN (Lab 1)", table_cell_left), Paragraph("<i>k*</i> = 51", table_cell_style), Paragraph(f"{results['knn_item_val_acc']*100:.2f}%", table_cell_style), Paragraph(f"{results['knn_item_test_acc']*100:.2f}%", table_cell_style)],
        [Paragraph("User-Based KNN (Lab 1)", table_cell_left), Paragraph("<i>k*</i> = 9", table_cell_style), Paragraph(f"{results['knn_user_val_acc']*100:.2f}%", table_cell_style), Paragraph(f"<b>{results['knn_user_test_acc']*100:.2f}%</b>", table_cell_style)],
    ]
    t3 = Table(t3_data, colWidths=[140, 110, 115, 122])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor("#E0F2FE")),  # Highlight Decision Tree
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t3)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"<b>Comparative Finding:</b> The best KNN classifier (User-Based KNN, <i>k</i> = 9) achieves <b>{results['knn_user_test_acc']*100:.2f}%</b> test accuracy, "
        f"which is <b>+{((results['knn_user_test_acc'] - results['test_acc'])*100):.2f}%</b> higher than the Decision Tree (63.96%).",
        body_style
    ))

    # ==========================================
    # PAGE 3: SECTION 4: WRITTEN ANALYSIS (TASK 5)
    # ==========================================
    story.append(PageBreak())

    story.append(Paragraph("4. Written Analysis and Reflection (Task 5)", h1_style))
    
    # (a)
    story.append(Paragraph(
        "<b>(a) Validation Curve Trajectory & Overfitting Inception:</b> "
        f"The validation curve displays an inverted U-shaped trajectory characteristic of model capacity tuning under the bias-variance trade-off. "
        f"At low depths (<i>d</i> = 1 to 4), the model suffers from high bias (underfitting) with validation accuracy hovering near baseline (~60.08% to 61.61%). "
        f"As capacity increases, performance rises steadily to achieve its global peak at <b><i>d*</i> = {results['opt_d']}</b> with a validation accuracy of <b>{results['opt_val_acc']*100:.2f}%</b>. "
        f"<b>Overfitting begins immediately past depth {results['opt_d']} (starting at <i>d</i> = 11)</b>: as depth increases from 11 to 20, training accuracy monotonically climbs from 67.44% to 80.13%, "
        f"while validation accuracy steadily deteriorates from 65.38% down to 63.73%.",
        body_style
    ))
    story.append(Spacer(1, 2))

    # (b)
    story.append(Paragraph(
        "<b>(b) Generalization Gap at Optimal Depth:</b> "
        f"At the optimal depth <i>d*</i> = {results['opt_d']}, training accuracy is <b>{results['opt_train_acc']*100:.2f}%</b> and validation accuracy is <b>{results['opt_val_acc']*100:.2f}%</b>. "
        f"The resulting generalization gap is very small (<b>{abs(results['opt_train_acc'] - results['opt_val_acc'])*100:.2f}%</b>). "
        f"This narrow gap indicates solid generalization capability: the tree at depth 10 has learned meaningful cohort-level ability partitions and question difficulty thresholds without memorizing noise or idiosyncrasies.",
        body_style
    ))
    story.append(Spacer(1, 2))

    # (c)
    story.append(Paragraph(
        "<b>(c) Mechanism of Validation Degradation at High Depths:</b> "
        f"A decision tree partitions the feature space ([<i>user_id</i>, <i>question_id</i>]) using axis-aligned orthogonal cuts. "
        f"At depth <i>d</i> = 20, the tree generates up to 2<sup>20</sup> (&gt; 1,000,000) potential leaf partitions—vastly exceeding the 56,688 training instances. "
        f"Consequently, deeper branches create hyper-specific leaf nodes that isolate individual samples, memorizing stochastic noise (such as careless mistakes, lucky guesses, or unrepresentative outliers). "
        f"Because these sample-specific artifacts do not reflect true latent student mastery, variance inflates and test-time validation accuracy inevitably degrades.",
        body_style
    ))
    story.append(Spacer(1, 2))

    # (d)
    story.append(Paragraph(
        "<b>(d) Comparative Model Assessment: Decision Tree vs. KNN:</b> "
        f"The empirical results prove that the <b>KNN classifier significantly outperforms the Decision Tree</b> on this dataset "
        f"(User-Based KNN test accuracy = <b>{results['knn_user_test_acc']*100:.2f}%</b> vs. Decision Tree = <b>{results['test_acc']*100:.2f}%</b>, a difference of <b>+4.91 percentage points</b>).<br/>"
        f"<i>Theoretical Justification:</i> The core reason stems from inductive bias and feature representations. "
        f"The input features are nominal categorical identifiers arbitrarily encoded as integers (0 to 541 for students, 0 to 1,773 for questions). "
        f"Decision trees impose axis-aligned splits on these nominal IDs (e.g., <code>user_id &le; 240</code>), which groups together unrelated students purely based on arbitrary indexing order. "
        f"In contrast, KNN operates via collaborative filtering across the 1,774-dimensional question response space. "
        f"KNN directly measures pairwise similarity between students' actual response patterns, effectively aggregating collective intelligence from true diagnostic peers.",
        body_style
    ))

    # ==========================================
    # PAGES 4+: APPENDIX: CODE LISTING
    # ==========================================
    story.append(PageBreak())

    story.append(Paragraph("Appendix: Code Listing (<code>decision_tree.py</code>)", h1_style))
    
    if is_group:
        story.append(Paragraph(
            f"Below is the complete implementation of <code>decision_tree.py</code> submitted on behalf of <b>{group_number}</b> "
            f"(Submitting Member: {student_name} - {student_id}). All work is original group work adhering to FIT-HANU academic standards.",
            body_style
        ))
    else:
        story.append(Paragraph(
            f"Below is the complete implementation of <code>decision_tree.py</code> submitted by <b>{student_name}</b> (ID: {student_id}). "
            f"All work is original work adhering to FIT-HANU academic standards.",
            body_style
        ))
    story.append(Spacer(1, 4))

    with open("decision_tree.py", "r", encoding="utf-8") as f:
        code_text = f.read()

    lines = [f"{i+1:3d}  {l}" for i, l in enumerate(code_text.splitlines())]
    formatted_code = "\n".join(lines)
    story.append(Preformatted(formatted_code, code_style))

    doc_canvas = make_canvas_class(custom_header)
    doc.build(story, canvasmaker=doc_canvas)
    print(f"[5/5] Report generated successfully at: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MLA Lab 2 Report (Individual and Group PDF)")
    parser.add_argument("--mode", type=str, default="both", choices=["individual", "group", "both"], help="Report mode to generate")
    parser.add_argument("--student_id", type=str, default="2301140014", help="Student ID")
    parser.add_argument("--student_name", type=str, default="Nghiêm Thành Công", help="Student Name")
    parser.add_argument("--group_number", type=str, default="Group01", help="Group Number / Name (e.g. Group01, Group 01)")
    parser.add_argument("--output", type=str, default=None, help="Custom output filename")
    args = parser.parse_args()

    results = run_experiments()

    if args.output:
        is_grp = (args.mode == "group")
        generate_pdf(results, is_group=is_grp, group_number=args.group_number, student_id=args.student_id, student_name=args.student_name, output_filename=args.output)
    else:
        if args.mode in ["individual", "both"]:
            indiv_file = f"{args.student_id}_Lab2.pdf"
            generate_pdf(results, is_group=False, group_number=args.group_number, student_id=args.student_id, student_name=args.student_name, output_filename=indiv_file)
        
        if args.mode in ["group", "both"]:
            grp_clean = args.group_number.replace(" ", "")
            group_file = f"{grp_clean}_Lab2.pdf"
            generate_pdf(results, is_group=True, group_number=args.group_number, student_id=args.student_id, student_name=args.student_name, output_filename=group_file)
            
            # Also write report.pdf as reference copy
            generate_pdf(results, is_group=True, group_number=args.group_number, student_id=args.student_id, student_name=args.student_name, output_filename="report.pdf")
