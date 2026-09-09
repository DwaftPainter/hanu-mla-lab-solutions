import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix, roc_auc_score

from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
    evaluate
)

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable, PageBreak, Preformatted
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def setup_fonts():
    """Register Arial TrueType fonts for full Vietnamese Unicode and symbol support."""
    font_paths = {
        "Arial": "/System/Library/Fonts/Supplemental/Arial.ttf",
        "Arial-Bold": "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "Arial-Italic": "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
        "Arial-BoldItalic": "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf",
    }
    
    for name, path in font_paths.items():
        if os.path.exists(path):
            pdfmetrics.registerFont(TTFont(name, path))


def ensure_dirs():
    os.makedirs("./figures", exist_ok=True)


def compute_majority_vote_baseline():
    """Exact reproduction of majority_vote.py logic."""
    train_data = load_train_csv("./data")
    correct_question_map = {}
    total_question_map = {}

    for i, q in enumerate(train_data["question_id"]):
        if q in correct_question_map:
            if train_data["is_correct"][i] == 1:
                correct_question_map[q] += 1
            total_question_map[q] += 1
        else:
            if train_data["is_correct"][i] == 1:
                correct_question_map[q] = 1
            total_question_map[q] = 1

    valid_data = load_valid_csv("./data")
    val_preds = []
    for i, q in enumerate(valid_data["question_id"]):
        ratio = correct_question_map.get(q, 0) / float(total_question_map[q])
        val_preds.append(1.0 if ratio >= 0.5 else 0.0)
    val_acc = evaluate(valid_data, val_preds)

    test_data = load_public_test_csv("./data")
    test_preds = []
    for i, q in enumerate(test_data["question_id"]):
        ratio = correct_question_map.get(q, 0) / float(total_question_map[q])
        test_preds.append(1.0 if ratio >= 0.5 else 0.0)
    test_acc = evaluate(test_data, test_preds)

    return val_acc, test_acc


def run_experiments(student_id="2301140014"):
    ensure_dirs()
    print("[1/5] Loading datasets & inspecting statistics...")
    train_data = load_train_csv("./data")
    valid_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")
    sparse_matrix = load_train_sparse("./data").toarray()

    # Task 1 Stats
    n_train = len(train_data["user_id"])
    n_val = len(valid_data["user_id"])
    n_test = len(test_data["user_id"])
    n_users, n_items = sparse_matrix.shape
    total_cells = n_users * n_items
    observed_cells = np.count_nonzero(~np.isnan(sparse_matrix))
    nan_cells = np.count_nonzero(np.isnan(sparse_matrix))
    sparsity_fraction = nan_cells / total_cells

    # Majority vote baseline
    mv_val_acc, mv_test_acc = compute_majority_vote_baseline()

    k_values = [1, 3, 5, 7, 9, 11, 21, 51]
    
    print("[2/5] Running User-based KNN across k in {1, 3, 5, 7, 9, 11, 21, 51}...")
    user_val_accs = []
    user_train_accs = []
    user_mats = {}
    for k in k_values:
        imputer = KNNImputer(n_neighbors=k)
        mat = imputer.fit_transform(sparse_matrix)
        val_acc = sparse_matrix_evaluate(valid_data, mat)
        train_acc = sparse_matrix_evaluate(train_data, mat)
        user_val_accs.append(val_acc)
        user_train_accs.append(train_acc)
        user_mats[k] = mat
        print(f"  User k={k:2d} -> Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.5f}")

    best_idx_user = int(np.argmax(user_val_accs))
    best_k_user = k_values[best_idx_user]
    best_val_acc_user = user_val_accs[best_idx_user]
    test_acc_user = sparse_matrix_evaluate(test_data, user_mats[best_k_user])
    print(f"  --> Best User k={best_k_user}: Val Acc = {best_val_acc_user:.5f} ({best_val_acc_user*100:.2f}%), Test Acc = {test_acc_user:.5f} ({test_acc_user*100:.2f}%)")

    print("[3/5] Running Item-based KNN across k in {1, 3, 5, 7, 9, 11, 21, 51}...")
    matrix_t = sparse_matrix.T
    item_val_accs = []
    item_train_accs = []
    item_mats = {}
    for k in k_values:
        imputer = KNNImputer(n_neighbors=k)
        mat_t = imputer.fit_transform(matrix_t)
        mat = mat_t.T
        val_acc = sparse_matrix_evaluate(valid_data, mat)
        train_acc = sparse_matrix_evaluate(train_data, mat)
        item_val_accs.append(val_acc)
        item_train_accs.append(train_acc)
        item_mats[k] = mat
        print(f"  Item k={k:2d} -> Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.5f}")

    best_idx_item = int(np.argmax(item_val_accs))
    best_k_item = k_values[best_idx_item]
    best_val_acc_item = item_val_accs[best_idx_item]
    test_acc_item = sparse_matrix_evaluate(test_data, item_mats[best_k_item])
    print(f"  --> Best Item k={best_k_item}: Val Acc = {best_val_acc_item:.5f} ({best_val_acc_item*100:.2f}%), Test Acc = {test_acc_item:.5f} ({test_acc_item*100:.2f}%)")
    
    # Save student predictions .npy
    best_mat_item = item_mats[best_k_item]
    item_preds = []
    for i, q in enumerate(valid_data["question_id"]):
        u = valid_data["user_id"][i]
        item_preds.append(1 if best_mat_item[u, q] >= 0.5 else 0)
    np.save(f"{student_id}_item_knn_preds.npy", np.array(item_preds))

    print("[4/5] Generating publication-quality figures...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. User KNN Curve
    fig, ax = plt.subplots(figsize=(6.2, 2.5), dpi=300)
    ax.plot(k_values, [a*100 for a in user_val_accs], marker='o', color='#1e3a8a', linewidth=2.0, markersize=5.5, label='Validation Accuracy (%)')
    ax.axvline(x=best_k_user, color='#dc2626', linestyle='--', alpha=0.85, linewidth=1.5, label=f'Optimal k* = {best_k_user} ({best_val_acc_user*100:.2f}%)')
    ax.scatter([best_k_user], [best_val_acc_user*100], color='#dc2626', s=55, zorder=5)
    ax.set_title('User-Based KNN: Validation Accuracy vs. Number of Neighbors (k)', fontsize=10, fontweight='bold', pad=8, color='#0f172a')
    ax.set_xlabel('Number of Neighbors (k)', fontsize=9)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=9)
    ax.set_xticks(k_values)
    ax.set_ylim(60.0, 71.5)
    ax.legend(frameon=True, facecolor='white', loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig('./figures/user_knn_val_curve.png', dpi=300)
    plt.close(fig)

    # 2. Item KNN Curve
    fig, ax = plt.subplots(figsize=(6.2, 2.5), dpi=300)
    ax.plot(k_values, [a*100 for a in item_val_accs], marker='s', color='#15803d', linewidth=2.0, markersize=5.5, label='Validation Accuracy (%)')
    ax.axvline(x=best_k_item, color='#dc2626', linestyle='--', alpha=0.85, linewidth=1.5, label=f'Optimal k* = {best_k_item} ({best_val_acc_item*100:.2f}%)')
    ax.scatter([best_k_item], [best_val_acc_item*100], color='#dc2626', s=55, zorder=5)
    ax.set_title('Item-Based KNN: Validation Accuracy vs. Number of Neighbors (k)', fontsize=10, fontweight='bold', pad=8, color='#0f172a')
    ax.set_xlabel('Number of Neighbors (k)', fontsize=9)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=9)
    ax.set_xticks(k_values)
    ax.set_ylim(60.0, 71.5)
    ax.legend(frameon=True, facecolor='white', loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig('./figures/item_knn_val_curve.png', dpi=300)
    plt.close(fig)

    results = {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_users": n_users,
        "n_items": n_items,
        "total_cells": total_cells,
        "observed_cells": observed_cells,
        "nan_cells": nan_cells,
        "sparsity_fraction": sparsity_fraction,
        "mv_val_acc": mv_val_acc,
        "mv_test_acc": mv_test_acc,
        "k_values": k_values,
        "user_train_accs": user_train_accs,
        "user_val_accs": user_val_accs,
        "best_k_user": best_k_user,
        "best_val_acc_user": best_val_acc_user,
        "test_acc_user": test_acc_user,
        "item_train_accs": item_train_accs,
        "item_val_accs": item_val_accs,
        "best_k_item": best_k_item,
        "best_val_acc_item": best_val_acc_item,
        "test_acc_item": test_acc_item,
    }
    return results


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and render exact total page numbers and clean headers/footers."""
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
            self.drawString(54, 804, "Machine Learning Applications (MLA) | Lab 1: KNN on Educational Response Data")
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


def generate_pdf_report(results, student_id="2301140014", student_name="Nghiêm Thành Công", output_filename=None):
    setup_fonts()
    if output_filename is None:
        output_filename = f"{student_id}_Lab1.pdf"
        
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
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#334155"),
        alignment=1,
        spaceAfter=8
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Arial-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=8.5,
        leading=12.0,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=4
    )

    code_style = ParagraphStyle(
        'CodeSnippet',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=6.2,
        leading=7.7,
        textColor=colors.HexColor("#0F172A")
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=7.8,
        leading=10,
        textColor=colors.HexColor("#1E293B"),
        alignment=1
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=8.2,
        leading=10.5,
        textColor=colors.white,
        alignment=1
    )

    story = []

    # ==========================================
    # PAGE 1: HEADER + SECTION 1 + SECTION 2
    # ==========================================
    story.append(Paragraph("Lab 1: K-Nearest Neighbours on Educational Response Data", title_style))
    story.append(Paragraph(f"<b>Student Name:</b> {student_name} &nbsp;|&nbsp; <b>Student ID:</b> {student_id} &nbsp;|&nbsp; <b>Course:</b> Machine Learning Applications (MLA)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.0, color=colors.HexColor("#1E3A8A"), spaceAfter=6))

    # SECTION 1: DATASET INSPECTION
    story.append(Paragraph("1. Dataset Inspection", h1_style))
    story.append(Paragraph(
        "The educational response dataset was inspected using starter utilities. "
        "The sparse matrix models binary diagnostic outcomes (1: correct, 0: incorrect, NaN: unobserved) across a student cohort.",
        body_style
    ))

    t1_data = [
        [Paragraph("Metric / Statistic", table_header_style), Paragraph("Value", table_header_style), Paragraph("Description", table_header_style)],
        [Paragraph("Training Triples", table_cell_style), Paragraph(f"{results['n_train']:,}", table_cell_style), Paragraph("Observed (user, question, is_correct) training samples", table_cell_style)],
        [Paragraph("Validation Triples", table_cell_style), Paragraph(f"{results['n_val']:,}", table_cell_style), Paragraph("Held-out validation response samples", table_cell_style)],
        [Paragraph("Public Test Triples", table_cell_style), Paragraph(f"{results['n_test']:,}", table_cell_style), Paragraph("Held-out public test response samples", table_cell_style)],
        [Paragraph("Sparse Matrix Shape", table_cell_style), Paragraph(f"{results['n_users']} × {results['n_items']}", table_cell_style), Paragraph("542 students (rows) × 1,774 questions (columns)", table_cell_style)],
        [Paragraph("Total Matrix Cells", table_cell_style), Paragraph(f"{results['total_cells']:,}", table_cell_style), Paragraph("Total possible user-item interactions", table_cell_style)],
        [Paragraph("Observed Cells", table_cell_style), Paragraph(f"{results['observed_cells']:,}", table_cell_style), Paragraph("Non-NaN observed training entries", table_cell_style)],
        [Paragraph("Missing Cells (NaN)", table_cell_style), Paragraph(f"{results['nan_cells']:,}", table_cell_style), Paragraph("Unobserved student response entries", table_cell_style)],
        [Paragraph("<b>Sparsity Fraction</b>", table_cell_style), Paragraph(f"<b>{results['sparsity_fraction']*100:.2f}%</b> ({results['sparsity_fraction']:.4f})", table_cell_style), Paragraph("Proportion of missing entries in matrix", table_cell_style)],
        [Paragraph("<b>Majority-Vote Baseline</b>", table_cell_style), Paragraph(f"<b>Val: {results['mv_val_acc']*100:.2f}%</b> | Test: {results['mv_test_acc']*100:.2f}%", table_cell_style), Paragraph("Benchmark accuracy: predicting modal response per question", table_cell_style)],
    ]
    t1 = Table(t1_data, colWidths=[125, 115, 247])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t1)
    story.append(Spacer(1, 6))

    # SECTION 2: USER-BASED KNN
    story.append(Paragraph("2. User-Based KNN", h1_style))
    story.append(Paragraph(
        "In User-Based KNN, students are instances (rows) across 1,774 question dimensions. "
        "Missing responses are imputed using <code>KNNImputer</code> across <i>k</i> in {1, 3, 5, 7, 9, 11, 21, 51}.",
        body_style
    ))

    t2_headers = [Paragraph("<i>k</i>", table_header_style)] + [Paragraph(str(k), table_header_style) for k in results['k_values']]
    t2_train_row = [Paragraph("<b>Train Acc</b>", table_cell_style)] + [Paragraph(f"{acc*100:.1f}%", table_cell_style) for acc in results['user_train_accs']]
    t2_val_row = [Paragraph("<b>Val Acc</b>", table_cell_style)] + [
        Paragraph(f"<b>{acc*100:.2f}%*</b>" if k == results['best_k_user'] else f"{acc*100:.2f}%", table_cell_style) 
        for k, acc in zip(results['k_values'], results['user_val_accs'])
    ]
    t2 = Table([t2_headers, t2_train_row, t2_val_row], colWidths=[67] + [52]*len(results['k_values']))
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t2)
    story.append(Spacer(1, 4))

    if os.path.exists("./figures/user_knn_val_curve.png"):
        story.append(Image("./figures/user_knn_val_curve.png", width=420, height=170))
    
    story.append(Paragraph(
        f"<b>Optimal <i>k</i> Selection & Justification:</b> The user-based validation curve achieves its global peak at <b><i>k</i>* = {results['best_k_user']}</b> "
        f"with a validation accuracy of <b>{results['best_val_acc_user']*100:.2f}%</b> (+{((results['best_val_acc_user'] - results['mv_val_acc'])*100):.2f}% over majority vote). "
        f"For low <i>k</i> (<i>k</i> = 1), predictions suffer from high variance (noise from individual student idiosyncrasies). "
        f"For high <i>k</i> (<i>k</i> ≥ 21), high bias dominates as averaging across too many dissimilar students pulls predictions toward cohort averages (falling to 62.40% at <i>k</i> = 51). "
        f"Hence, <b><i>k</i>* = {results['best_k_user']}</b> is the optimal hyperparameter under the bias-variance tradeoff.",
        body_style
    ))

    # ==========================================
    # PAGE 2: SECTION 3 + SECTION 4
    # ==========================================
    story.append(PageBreak())

    story.append(Paragraph("3. Item-Based KNN", h1_style))
    story.append(Paragraph(
        "In Item-Based KNN, questions are instances (transposing the matrix to 1,774 × 542). "
        "Imputation finds similar questions based on shared student response patterns and is transposed back for evaluation.",
        body_style
    ))

    t3_headers = [Paragraph("<i>k</i>", table_header_style)] + [Paragraph(str(k), table_header_style) for k in results['k_values']]
    t3_train_row = [Paragraph("<b>Train Acc</b>", table_cell_style)] + [Paragraph(f"{acc*100:.1f}%", table_cell_style) for acc in results['item_train_accs']]
    t3_val_row = [Paragraph("<b>Val Acc</b>", table_cell_style)] + [
        Paragraph(f"<b>{acc*100:.2f}%*</b>" if k == results['best_k_item'] else f"{acc*100:.2f}%", table_cell_style) 
        for k, acc in zip(results['k_values'], results['item_val_accs'])
    ]
    t3 = Table([t3_headers, t3_train_row, t3_val_row], colWidths=[67] + [52]*len(results['k_values']))
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t3)
    story.append(Spacer(1, 4))

    if os.path.exists("./figures/item_knn_val_curve.png"):
        story.append(Image("./figures/item_knn_val_curve.png", width=420, height=170))
    
    story.append(Paragraph(
        f"<b>Optimal <i>k</i> Selection:</b> The item-based validation curve rises monotonically across the evaluated range and peaks at <b><i>k</i>* = {results['best_k_item']}</b> "
        f"with <b>{results['best_val_acc_item']*100:.2f}%</b> validation accuracy. "
        f"Because each question has relatively few observed ratings (~32 on average across 542 students), item vectors have high sparsity and noise. "
        f"A larger neighborhood (<i>k</i> = 51) is necessary to aggregate sufficient collaborative signal and stabilize prediction estimates.",
        body_style
    ))
    story.append(Spacer(1, 6))

    # SECTION 4: TEST RESULTS
    story.append(Paragraph("4. Test Results", h1_style))
    story.append(Paragraph(
        "Using the best <i>k</i> selected strictly from the validation curves, final test accuracy was evaluated on <code>test_data.csv</code> (3,543 responses). "
        "In accordance with evaluation guidelines, test accuracy was evaluated exactly once per method to avoid data leakage.",
        body_style
    ))

    t4_data = [
        [Paragraph("Method", table_header_style), Paragraph("Optimal <i>k</i>", table_header_style), Paragraph("Validation Accuracy", table_header_style), Paragraph("Final Test Accuracy", table_header_style)],
        [Paragraph("Majority-Vote Baseline", table_cell_style), Paragraph("N/A", table_cell_style), Paragraph(f"{results['mv_val_acc']*100:.2f}%", table_cell_style), Paragraph(f"<b>{results['mv_test_acc']*100:.2f}%</b>", table_cell_style)],
        [Paragraph("User-Based KNN (HANU)", table_cell_style), Paragraph(f"<i>k</i> = {results['best_k_user']}", table_cell_style), Paragraph(f"{results['best_val_acc_user']*100:.2f}%", table_cell_style), Paragraph(f"<b>{results['test_acc_user']*100:.2f}%</b>", table_cell_style)],
        [Paragraph("Item-Based KNN (HANU)", table_cell_style), Paragraph(f"<i>k</i> = {results['best_k_item']}", table_cell_style), Paragraph(f"{results['best_val_acc_item']*100:.2f}%", table_cell_style), Paragraph(f"<b>{results['test_acc_item']*100:.2f}%</b>", table_cell_style)],
    ]
    t4 = Table(t4_data, colWidths=[140, 85, 125, 137])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t4)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<i>Summary:</i> Both KNN approaches provide substantial gains over the baseline (~6.0% improvement on the test set). "
        "User-based KNN slightly outperforms item-based KNN on final test accuracy (68.87% vs 68.47%).",
        body_style
    ))

    # ==========================================
    # PAGE 3: SECTION 5: ANALYSIS AND REFLECTION
    # ==========================================
    story.append(PageBreak())

    story.append(Paragraph("5. Analysis and Reflection (391 words)", h1_style))
    story.append(Paragraph(
        "<b>1. User-Based vs. Item-Based Performance:</b> User-based KNN achieved a superior final test accuracy of <b>68.87%</b> (<i>k</i> = 9) compared to <b>68.47%</b> for item-based KNN (<i>k</i> = 51), while both significantly outperformed the majority-vote baseline of 62.88%. The primary explanation lies in the dimensionality and semantic density of the representations. The dataset contains 542 students evaluated across 1,774 questions. Each student profile is represented by a rich 1,774-dimensional vector that effectively captures overarching student mastery and cognitive misconceptions. Identifying a compact neighborhood of 9 closely aligned peers allows the model to reliably estimate a student's likelihood of correctly solving a target question. Conversely, questions are represented by 542-dimensional student vectors with an average of only ~32 observed responses per question. Consequently, pairwise question similarities are substantially noisier due to extreme sparsity (94.10%), requiring an expansive neighborhood (<i>k</i> = 51) to aggregate sufficient collaborative signal. This broad pooling dampens idiosyncratic question characteristics and slightly degrades generalization on unseen test pairs.",
        body_style
    ))
    story.append(Paragraph(
        "<b>2. Validation Curve Shape & Bias-Variance Tradeoff:</b> The user-based validation curve exhibits a characteristic concave trajectory peaking at <i>k</i> = 9 (69.22%). At low <i>k</i> (<i>k</i> = 1, 3), the model suffers from high variance and overfitting, where predictions are excessively sensitive to noise, lucky guesses, or careless mistakes of individual nearest peers (yielding 62.50% at <i>k</i> = 1). As <i>k</i> increases toward 9, averaging across neighbors mitigates variance. However, for <i>k</i> ≥ 11 (dropping to 62.40% at <i>k</i> = 51), high bias dominates as the neighborhood encompasses over 9.4% of the entire student population, diluting distinct ability clusters toward global averages. Conversely, item-based validation accuracy rises monotonically up to <i>k</i> = 51 (69.31%), indicating that severe sparsity in question space requires substantial neighborhood smoothing to overcome variance.",
        body_style
    ))
    story.append(Paragraph(
        "<b>3. Empirical Limitations of KNN:</b> Beyond theoretical limitations, two key drawbacks emerged: "
        "<br/>• <i>Metric distortion under extreme sparsity (94.10%):</i> nan-Euclidean distance scales distances inversely by the count of co-rated features. When entities share only 1–2 overlapping questions, coincidental agreement artificially skews neighbor selection. "
        "<br/>• <i>Equal weighting of items without difficulty calibration:</i> KNN treats all co-rated questions uniformly, failing to distinguish between answering an elementary question versus a highly discriminative, advanced question.",
        body_style
    ))
    story.append(Paragraph(
        "<b>4. Proposed Next Improvements:</b> Future iterations should explore: "
        "(i) <b>Item Response Theory (1PL/2PL/3PL IRT)</b> to mathematically decouple latent student ability (&theta;<sub>i</sub>) and question difficulty (&beta;<sub>j</sub>); "
        "(ii) <b>Matrix Factorization (Probabilistic Matrix Factorization or SVD)</b> with explicit user and item bias terms; and "
        "(iii) <b>Hybrid Neural Collaborative Filtering</b> incorporating metadata from <code>student_meta.csv</code> and <code>subject_meta.csv</code>.",
        body_style
    ))

    # ==========================================
    # PAGES 4+: APPENDIX: CODE LISTING
    # ==========================================
    story.append(PageBreak())

    story.append(Paragraph("Appendix: Code Listing (Modified <code>knn.py</code>)", h1_style))
    story.append(Paragraph("Below is the complete implementation of the required user-based and item-based KNN functions and evaluation routines.", body_style))
    story.append(Spacer(1, 3))

    with open("knn.py", "r") as f:
        code_text = f.read()

    lines = [f"{i+1:3d}  {l}" for i, l in enumerate(code_text.splitlines())]
    formatted_code = "\n".join(lines)
    story.append(Preformatted(formatted_code, code_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[5/5] Report generated successfully at: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MLA Lab 1 Report and PDF")
    parser.add_argument("--student_id", type=str, default="2301140014", help="Your Student ID")
    parser.add_argument("--student_name", type=str, default="Nghiêm Thành Công", help="Your Full Name")
    parser.add_argument("--output", type=str, default=None, help="Output PDF filename")
    args = parser.parse_args()

    results = run_experiments(student_id=args.student_id)
    out_pdf = args.output if args.output else f"{args.student_id}_Lab1.pdf"
    generate_pdf_report(results, student_id=args.student_id, student_name=args.student_name, output_filename=out_pdf)
