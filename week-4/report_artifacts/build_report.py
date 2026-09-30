"""Assemble the supplied, executed Week 4 notebook into one group report."""
import json
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, Preformatted, KeepTogether

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
cfg = json.loads((ROOT / 'mla-report-config.json').read_text())
nb = json.loads((HERE / 'executed_notebook.ipynb').read_text())
env = json.loads((HERE / 'environment.json').read_text())
for name, filename in [('Arial', 'Arial.ttf'), ('Arial-Bold', 'Arial Bold.ttf')]:
    pdfmetrics.registerFont(TTFont(name, '/System/Library/Fonts/Supplemental/' + filename))
navy = colors.HexColor('#17365D')
body = ParagraphStyle('Body', fontName='Arial', fontSize=10, leading=14, spaceAfter=9)
heading = ParagraphStyle('Heading', parent=body, fontName='Arial-Bold', fontSize=15, leading=20, textColor=navy, spaceBefore=8, spaceAfter=12)
title = ParagraphStyle('Title', parent=heading, fontSize=24, leading=29)
small = ParagraphStyle('Small', parent=body, fontSize=8, leading=11, textColor=colors.HexColor('#475569'))
code = ParagraphStyle('Code', fontName='Courier', fontSize=7.1, leading=9.2, spaceAfter=8)
story = []

def p(text, style=body):
    story.append(Paragraph(escape(text), style))

def h(text):
    p(text, heading)

def table(rows, widths=None):
    cells = [[Paragraph(escape(str(v)), body) for v in row] for row in rows]
    t = Table(cells, colWidths=widths, hAlign='LEFT', repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EEF5')), ('VALIGN',(0,0),(-1,-1),'TOP'), ('LINEBELOW',(0,0),(-1,0),0.7,navy), ('BOTTOMPADDING',(0,0),(-1,-1),6), ('TOPPADDING',(0,0),(-1,-1),6), ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F6F8FA')])]))
    story.extend([t, Spacer(1,10)])

def output(index):
    for o in nb['cells'][index]['outputs']:
        if o['output_type']=='stream':
            s = o['text']
            s = ''.join(s) if isinstance(s,list) else s
            story.append(Preformatted(s.strip(), code))

def fig(index, caption, width=410):
    path = next(HERE.glob(f'cell_{index}_figure_*.png'))
    im = Image(str(path))
    im.drawHeight *= width / im.drawWidth
    im.drawWidth = width
    story.append(KeepTogether([im, Paragraph(escape(caption), small)]))

def page():
    story.append(PageBreak())

p('Machine Learning Applications', small)
p('Lab 4\nLogistic Regression and Gradient Descent'.replace('\n', ' — '), title)
table([['Submission', 'Details'], ['Group / class', cfg['group_prefix']], ['Submitting member', cfg['submitting_member']['name']], ['Student ID', cfg['submitting_member']['student_id']], ['Institution / term', 'Hanoi University · Semester 1, AY 2026–2027']], [145,342])
h('Dataset and execution setup')
p('The lab implements logistic regression from first principles and applies it to synthetic student–question attempts. The target is whether an answer is correct (1) or wrong (0). The report follows Parts A–I of the supplied completed Week 4 notebook; no separate lab instruction PDF was issued this week, as confirmed by the submitting member.')
p('Source: MLA_Week4_LogisticRegression_Lab_Solved.ipynb. All 11 code cells were executed sequentially in a fresh namespace; their current outputs and five plots are used here. The complete code is included in the appendix. The supplied notebook discloses AI assistance in its preparation; that disclosure is retained here.')
p('The dataset contains 4,000 generated attempts, with normally distributed ability, difficulty and practice features. The generating logit is 1.5 × ability − 1.5 × difficulty + 0.5 × practice + 1.3. A fixed seed of 42 produces a random 60/20/20 split: 2,400 training, 800 validation and 800 test attempts. No real Eedi response CSV is used.')
p('Environment: Python '+env['python']+'; NumPy '+env['numpy']+'; Matplotlib '+env['matplotlib']+'; scikit-learn '+env['scikit-learn']+'. Figures use the noninteractive Agg backend.', small)
page()
h('Part A — Sigmoid and cross-entropy')
p('For x = [−2, −1, 1, 2], targets t = [0, 0, 1, 1] and weights [0, 1], the model computes z = Xw and p = 1 / (1 + exp(−z)). Cross-entropy is the mean of −t log(p) − (1 − t) log(1 − p). The implementation uses a stable sigmoid and clips probabilities at the logarithm boundaries.')
output(5)
p('Probabilities are low for the two wrong answers and high for the two correct answers. A 0.5 threshold classifies all four correctly, but the nonzero loss still rewards more confident correct probabilities.')
h('Part B — One gradient step')
p('The average gradient is Xᵀ(p − t) / N. The residuals are approximately [0.1192, 0.2689, −0.2689, −0.1192]; their mean is zero. The slope component is (−2 × 0.1192 − 0.2689 − 0.2689 − 2 × 0.1192) / 4 ≈ −0.2537. With learning rate 0.5, w ← w − 0.5 × gradient.')
output(8)
p('The slope increases from 1 to approximately 1.1268, sharpening the sigmoid and reducing the cost from 0.2201 to 0.1903.')
h('Part C — Synthetic Eedi-style data')
output(11)
p('Each feature is known before the generated attempt. The intercept creates a majority of correct responses. The design matrix includes an explicit bias column, followed by ability, difficulty and practice. Training uses only the training partition; validation provides a library cross-check and the held-out test partition provides the final evaluation.')
page()
h('Part D — Full-batch gradient descent')
p('Training starts from zero weights and applies one full-training-set gradient update per epoch, using a learning rate of 0.3 for 300 epochs. The recorded cost is measured after each update.')
output(13)
fig(13, 'Figure 1. Full-batch training cross-entropy over 300 epochs.')
p('The cost decreases and flattens. The fitted coefficient signs match the generating rule: ability and practice increase the modeled chance of a correct answer, while difficulty decreases it. The coefficients need not equal the generating weights exactly because the outcomes were randomly sampled.')
h('Evaluation protocol')
p('The final classifier throughout this report is the prescribed 300-epoch full-batch model. The batch-size and learning-rate experiments illustrate optimization behavior; their training losses are not used to claim a test-selected optimum. Part G below reports the separate validation comparison.')
page()
h('Part E — Batch, mini-batch and stochastic updates')
p('All three methods use learning rate 0.3, 40 epochs and shuffle seed 0. An epoch visits every training example once; the number of parameter updates differs substantially.')
output(16)
fig(16, 'Figure 2. Training loss at the end of each epoch for three batch sizes.')
p('Mini-batch training reaches a low loss within this epoch budget, while SGD fluctuates with the fixed step size. Full batch makes only 40 updates in this experiment. These curves compare equal epochs, not equal update counts or wall-clock time; they do not establish a timing advantage.')
page()
h('Part F — Learning-rate sweep')
p('The full-batch model is trained for 120 epochs at each listed learning rate. These runs use the same training partition and zero initialization.')
output(18)
fig(18, 'Figure 3. Full-batch training loss across five learning rates.')
p('A learning rate of 0.01 converges slowly within 120 epochs. Rates of 1 and 3 reach approximately the same low training loss, whereas 30 produces unstable behavior and a much higher final loss. A rate of 3 is not inherently excessive; stability depends on the data and objective curvature.')
page()
h('Part G — Cross-check with scikit-learn')
p('The library model uses C=np.inf, fit_intercept=False, max_iter=1000 and tol=1e-10. Disabling its intercept avoids duplicating the bias column already present in X. Both models fit the same training partition.')
output(21)
p('The validation accuracies are close, and the fitted coefficients have similar signs and magnitudes. The 300-epoch from-scratch model has not exactly reached the library solution; the comparison supports consistency without asserting identical coefficients.')
h('Part H — Held-out test evaluation')
p('The positive class is “answered correctly”. The final model is evaluated at the fixed probability threshold of 0.5. Accuracy = (TP + TN)/N; precision = TP/(TP + FP); recall = TP/(TP + FN); F1 = 2TP/(2TP + FP + FN).')
output(23)
fig(23, 'Figure 4. Test confusion matrix; rows are actual labels and columns are predictions.', width=255)
page()
h('Part H continued — Baseline and thresholds')
p('The always-correct baseline achieves 69.25% test accuracy but catches none of the wrong answers. Logistic regression achieves 83.13% accuracy and identifies 169 of 246 wrong answers (68.70% wrong-answer recall). Its 58 false wrong-answer flags concern responses that were actually correct; these would need human review in a teacher-support setting.')
fig(26, 'Figure 5. Descriptive test-set precision and recall for the correct-answer class.')
p('For the positive class, lowering the threshold predicts more correct answers and cannot reduce recall. Precision is not guaranteed to change monotonically in a finite sample. For flagging wrong answers, raising the cutoff p(correct) < threshold sends more attempts for review, including more false alarms. This test-set sweep is descriptive only; choose an operational threshold using validation data and evaluate the fixed choice on an untouched test set.')
h('Part I — Responsible-use reflection')
reflection = ''.join(nb['cells'][28]['outputs'][0]['text']).strip()
p(' '.join(reflection.split()))
h('Conclusion and limitations')
p('The supplied implementation connects sigmoid probabilities, cross-entropy and gradient descent, and reproduces a similar validation result to the library model. The synthetic data follow a logistic generating rule and use idealized features. These results do not establish accuracy, calibration or fairness on real students or real Eedi data. No real-data or calibration experiment was performed.')
page()
h('Appendix — Complete submitted code')
p('All 11 code cells from the supplied solved notebook appear below in execution order. Long lines wrap for display only. The source notebook and its implementation are unchanged.', small)
for i,c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    p(f'Notebook cell {i} · execution {c["execution_count"]}', small)
    lines=[]
    for line in ''.join(c['source']).splitlines():
        lines.extend(textwrap.wrap(line, width=108, subsequent_indent='    ', replace_whitespace=False, drop_whitespace=False) or [''])
    story.append(Preformatted('\n'.join(lines),code))

def decorate(can, doc):
    can.saveState()
    can.setFont('Arial',8)
    can.setFillColor(navy)
    can.drawString(54,810,'MLA · Lab 4 · Logistic Regression and Gradient Descent')
    can.setStrokeColor(colors.HexColor('#CBD5E1'))
    can.line(54,801,541,801)
    can.line(54,45,541,45)
    can.drawString(54,32,cfg['group_prefix']+' · HANU · AY 2026–2027')
    can.drawRightString(541,32,f'Page {doc.page}')
    can.restoreState()

dest = HERE / (cfg['group_prefix']+'_Lab4.pdf')
doc = SimpleDocTemplate(str(dest),pagesize=A4,rightMargin=54,leftMargin=54,topMargin=58,bottomMargin=59,title='Lab 4 — Logistic Regression and Gradient Descent',author=cfg['submitting_member']['name'])
doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
print(dest)
