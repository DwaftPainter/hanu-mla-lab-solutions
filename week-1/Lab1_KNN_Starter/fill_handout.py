import os
import pymupdf

def fill_handout():
    input_pdf = "MLA_Week1_Tutorial_Handout.pdf"
    output_pdf = "MLA_Week1_Tutorial_Handout_Filled.pdf"

    doc = pymupdf.open(input_pdf)
    blue_color = (0.05, 0.22, 0.65) # Deep Professional Blue
    red_color = (0.75, 0.1, 0.1)     # Red for checkboxes

    # =============================================================
    # PAGE 1
    # =============================================================
    p1 = doc[0]
    # Name & Student ID (underlines at y=233.9)
    p1.insert_text((105, 230), "Nghiêm Thành Công", fontsize=10, fontname="helv", color=blue_color)
    p1.insert_text((438, 230), "2301140014", fontsize=10, fontname="helv", color=blue_color)

    # A.1 Fraction of missing entries (underlines at y=523.5)
    p1.insert_text((175, 520), "5", fontsize=9.5, fontname="helv", color=blue_color)
    p1.insert_text((342, 520), "20", fontsize=9.5, fontname="helv", color=blue_color)
    p1.insert_text((468, 520), "5 / 20 = 1/4 = 0.25 (25%)", fontsize=8.5, fontname="helv", color=blue_color)

    # A.2 Checkpoint: Why might KNN struggle (underlines at y=587.0, y=605)
    p1.insert_text((74, 583), "High matrix sparsity means learners share very few co-rated items in common, making distance", fontsize=8.5, fontname="helv", color=blue_color)
    p1.insert_text((74, 601), "estimates noisy, unreliable, or undefined due to lack of overlapping observations.", fontsize=8.5, fontname="helv", color=blue_color)

    # =============================================================
    # PAGE 2
    # =============================================================
    p2 = doc[1]
    # B.1 Distance Table
    # (A, B)
    p2.insert_text((120, 100), "Shared: {q1, q2, q3} (3 items) | Disagreements at q2, q3 (2)", fontsize=8.5, fontname="helv", color=blue_color)
    p2.insert_text((512, 100), "2/3 ≈ 0.67", fontsize=9, fontname="helv", color=blue_color)
    
    # (A, C)
    p2.insert_text((120, 124), "Shared: {q1, q3, q4} (3 items) | Disagreement at q1 (1)", fontsize=8.5, fontname="helv", color=blue_color)
    p2.insert_text((512, 124), "1/3 ≈ 0.33", fontsize=9, fontname="helv", color=blue_color)

    # (A, D)
    p2.insert_text((120, 147), "Shared: {q1, q2, q4} (3 items) | Disagreements: None (0)", fontsize=8.5, fontname="helv", color=blue_color)
    p2.insert_text((512, 147), "0/3 = 0.00", fontsize=9, fontname="helv", color=blue_color)

    # (B, C)
    p2.insert_text((120, 171), "Shared: {q1, q3, q5} (3 items) | Disagreements at q1, q3, q5 (3)", fontsize=8.5, fontname="helv", color=blue_color)
    p2.insert_text((512, 171), "3/3 = 1.00", fontsize=9, fontname="helv", color=blue_color)

    # Working calculation in full (underlines at y=254.7, y=281.0)
    p2.insert_text((74, 251), "For Pair (A, B): Both learners attempted {q1, q2, q3}. A = (1, 0, 1) and B = (1, 1, 0).", fontsize=8.5, fontname="helv", color=blue_color)
    p2.insert_text((74, 277), "Disagreements occur at q2 (0 ≠ 1) and q3 (1 ≠ 0) → 2 differences out of 3 shared items.", fontsize=8.5, fontname="helv", color=blue_color)
    p2.insert_text((74, 295), "Therefore, dH(A, B) = 2 / 3 ≈ 0.67.", fontsize=8.5, fontname="helv", color=blue_color)

    # C.1 Distance from Student D to A, B, C
    # (D, A)
    p2.insert_text((120, 464), "Shared: {q1, q2, q4} (3 items) | Disagreements: 0", fontsize=8.5, fontname="helv", color=blue_color)
    p2.insert_text((512, 464), "0/3 = 0.00", fontsize=9, fontname="helv", color=blue_color)

    # (D, B)
    p2.insert_text((120, 487), "Shared: {q1, q2} (2 items) | Disagreement at q2 (1)", fontsize=8.5, fontname="helv", color=blue_color)
    p2.insert_text((512, 487), "1/2 = 0.50", fontsize=9, fontname="helv", color=blue_color)

    # (D, C)
    p2.insert_text((120, 511), "Shared: {q1, q4} (2 items) | Disagreement at q1 (1)", fontsize=8.5, fontname="helv", color=blue_color)
    p2.insert_text((512, 511), "1/2 = 0.50", fontsize=9, fontname="helv", color=blue_color)

    # C.2 Nearest neighbours (underline at y=596.8)
    p2.insert_text((176, 593), "Student A (dH = 0.00) and Student B [or Student C] (dH = 0.50)", fontsize=8.5, fontname="helv", color=blue_color)

    # C.3 Votes on q5 (underlines at y=663.2, y=682.7, y=702.2)
    p2.insert_text((226, 660), "? (missing)", fontsize=8.5, fontname="helv", color=blue_color)
    p2.insert_text((300, 660), "N/A (no vote)", fontsize=8.5, fontname="helv", color=blue_color)

    p2.insert_text((226, 679), "0 (Student B)", fontsize=8.5, fontname="helv", color=blue_color)
    p2.insert_text((300, 679), "Vote: 0 (or 1 for C)", fontsize=8.5, fontname="helv", color=blue_color)

    p2.insert_text((232, 699), "Tie (0 vs 1) → 1 (by standard classification threshold ≥ 0.5)", fontsize=8.2, fontname="helv", color=blue_color)

    # =============================================================
    # PAGE 3
    # =============================================================
    p3 = doc[2]
    # D.2 Compute dH(q5, qj) (underlines at y=284.3, 305.8, 327.3, 348.8)
    p3.insert_text((136, 281), "Shared on {B, C}: B=(0 vs 1), C=(1 vs 0) → 2 / 2 = 1.00", fontsize=8.5, fontname="helv", color=blue_color)
    p3.insert_text((136, 302), "Shared on {B}: B=(0 vs 1) → 1 / 1 = 1.00", fontsize=8.5, fontname="helv", color=blue_color)
    p3.insert_text((136, 324), "Shared on {B, C}: B=(0 vs 0), C=(1 vs 1) → 0 / 2 = 0.00", fontsize=8.5, fontname="helv", color=blue_color)
    p3.insert_text((136, 345), "Shared on {C}: C=(1 vs 1) → 0 / 1 = 0.00", fontsize=8.5, fontname="helv", color=blue_color)

    # D.3 Two most similar items (underlines at y=430.4, 450.0, 469.5)
    p3.insert_text((196, 427), "q3 (dH = 0.00) and q4 (dH = 0.00)", fontsize=8.5, fontname="helv", color=blue_color)
    p3.insert_text((190, 447), "q3 = ? (missing), q4 = 1 (correct)", fontsize=8.5, fontname="helv", color=blue_color)
    p3.insert_text((232, 466), "1 (Correct, determined by available neighbor q4 = 1)", fontsize=8.5, fontname="helv", color=blue_color)

    # E.1 Same prediction? (checkbox 'No' at x=203.2, y=573)
    p3.insert_text((204.5, 576.5), "X", fontsize=9, fontname="helv", color=red_color)
    p3.insert_text((420, 576), "Tie (0/1)", fontsize=8.5, fontname="helv", color=blue_color)
    p3.insert_text((142, 589.5), "1 (Correct)", fontsize=8.5, fontname="helv", color=blue_color)

    # E.2 Which method to trust more (underlines at y=670.0, y=696.3)
    p3.insert_text((74, 666), "Item-based KNN is more decisive here because items q3 and q4 have perfect similarity (dH = 0)", fontsize=8.5, fontname="helv", color=blue_color)
    p3.insert_text((74, 692), "with q5, whereas learner-based KNN's closest peer (Student A) had not attempted q5.", fontsize=8.5, fontname="helv", color=blue_color)
    p3.insert_text((74, 710), "Empirically, a held-out validation set is required to evaluate and compare accuracy across K.", fontsize=8.5, fontname="helv", color=blue_color)

    # =============================================================
    # PAGE 4
    # =============================================================
    p4 = doc[3]
    # E.3 Complete the sentences (underlines at y=80.2, 101.7, 123.3)
    p4.insert_text((326, 77), "learners answer disjoint item sets, yielding low overlap and noisy distances.", fontsize=8.2, fontname="helv", color=blue_color)
    p4.insert_text((319, 98), "items have few shared learners who attempted both, making similarities noisy.", fontsize=8.2, fontname="helv", color=blue_color)
    p4.insert_text((230, 120), "more response data were collected, or latent factor models (e.g. IRT / SVD) were used.", fontsize=8.2, fontname="helv", color=blue_color)

    # F.1 Checkboxes (boxes at x=63.7, y=254.8, 275.1, 295.4, 315.7)
    p4.insert_text((65.0, 263.0), "X", fontsize=8.5, fontname="helv", color=red_color)
    p4.insert_text((65.0, 283.3), "X", fontsize=8.5, fontname="helv", color=red_color)
    p4.insert_text((65.0, 303.6), "X", fontsize=8.5, fontname="helv", color=red_color)
    p4.insert_text((65.0, 323.9), "X", fontsize=8.5, fontname="helv", color=red_color)

    # F.2 Primary research question for Lab 1 (underlines at y=384.2, y=400)
    p4.insert_text((74, 380), "How do User-Based and Item-Based KNN compare in predicting student responses on a sparse educational", fontsize=8.5, fontname="helv", color=blue_color)
    p4.insert_text((74, 398), "dataset, and which value of K achieves the optimal validation accuracy under the bias-variance tradeoff?", fontsize=8.5, fontname="helv", color=blue_color)

    doc.save(output_pdf)
    print(f"Saved filled PDF to {output_pdf}")

if __name__ == "__main__":
    fill_handout()
