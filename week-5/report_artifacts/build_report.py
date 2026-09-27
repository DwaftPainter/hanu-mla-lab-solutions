"""Build one Week 5 group report from the executed solved notebook."""
import base64
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, Preformatted, KeepTogether,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE.parent / 'MLA_Week5_SVM_Regularisation_Lab_Solved.ipynb'
nb = json.loads(SOURCE.read_text())
cfg = json.loads((ROOT / 'mla-report-config.json').read_text())
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
assert all(c.get('execution_count') is not None for c in code_cells)
assert not any(o['output_type'] == 'error' for c in code_cells for o in c['outputs'])
for name, file in [('Arial', 'Arial.ttf'), ('Arial-Bold', 'Arial Bold.ttf')]:
    pdfmetrics.registerFont(TTFont(name, '/System/Library/Fonts/Supplemental/' + file))
pdfmetrics.registerFont(TTFont('Menlo', '/System/Library/Fonts/Menlo.ttc', subfontIndex=0))
navy = colors.HexColor('#17365D')
body = ParagraphStyle('Body', fontName='Arial', fontSize=10, leading=14, spaceAfter=9)
heading = ParagraphStyle('Heading', parent=body, fontName='Arial-Bold', fontSize=15,
                         leading=20, textColor=navy, spaceBefore=8, spaceAfter=12,
                         keepWithNext=True)
title = ParagraphStyle('Title', parent=heading, fontSize=24, leading=29)
small = ParagraphStyle('Small', parent=body, fontSize=8, leading=11, textColor='#475569')
code = ParagraphStyle('Code', fontName='Menlo', fontSize=7, leading=9.2, spaceAfter=9)
story = []

def p(text, style=body):
    story.append(Paragraph(escape(text), style))

def h(text):
    p(text, heading)

def page():
    story.append(PageBreak())

def table(rows, widths):
    t = Table([[Paragraph(escape(str(x)), body) for x in row] for row in rows],
              colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E8EEF5')),
        ('LINEBELOW', (0,0), (-1,0), .7, navy),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F6F8FA')]),
        ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.extend([t, Spacer(1, 10)])

def output(index):
    for o in nb['cells'][index].get('outputs', []):
        if o['output_type'] == 'stream':
            text = o['text'] if isinstance(o['text'], str) else ''.join(o['text'])
            story.append(Preformatted(text.strip(), code, maxLineLength=107))

def fig(index, caption, width=430):
    data = next(o['data']['image/png'] for o in nb['cells'][index]['outputs']
                if 'image/png' in o.get('data', {}))
    path = HERE / f'cell_{index}_figure.png'
    path.write_bytes(base64.b64decode(data))
    im = Image(str(path))
    im.drawHeight *= width / im.drawWidth
    im.drawWidth = width
    story.append(KeepTogether([im, Paragraph(escape(caption), small)]))

p('Machine Learning Applications', small)
p('Lab 5 — SVM and Regularization', title)
table([['Submission', 'Details'], ['Group / class', 'Group 05 / CLC02'],
       ['Submitting member', cfg['submitting_member']['name']],
       ['Student ID', cfg['submitting_member']['student_id']],
       ['Institution / term', 'Hanoi University · Semester 1, AY 2026–2027']], [155, 320])
h('Dataset and execution setup')
p('This report follows Parts A–J of the Week 5 notebook and uses the layout of the previous group reports, as requested by the submitting member. The tasks cover SVM margins, hinge loss, subgradient descent, regularization, validation, cross-validation, feature scaling and responsible interpretation.')
p('All 20 code cells were executed in order in a clean Jupyter kernel. The four figures and numeric outputs below come from that execution. The complete code is included in the appendix. The worked notebook was prepared with AI assistance; this report retains that disclosure.')
p('Parts A–B use five hand-specified points plus one noisy observation. Parts C–D use 400 synthetic observations in two overlapping classes. Parts E–J use 400 synthetic Eedi-style attempts with 20 features: four generating features and sixteen noise features. No real Eedi response CSV is used.')
p('Reproducibility: NumPy default_rng(42); Python 3.14.7; NumPy 2.5.3; scikit-learn 1.7.2. The notebook prints its execution environment. The starter’s reference Python version differs from this actual run.', small)
p('Source: ' + SOURCE.name, small)
page()
h('Part A — The margin, by hand')
p('For w = (1, 0) and b = −2, the score is x1 − 2, the functional margin is y × score, and the corridor width is 2 / ||w||. A positive margin means the point is correctly classified; margin at least one has zero hinge loss.')
output(5)
p('The four points at margin 1 lie on the corridor boundaries. The last point has margin 3 and lies beyond the corridor. Margin location alone does not determine whether a point has a nonzero dual coefficient.')
output(9)
p('The solver represents this hard-margin boundary using two of the four margin points; the others are redundant. In a soft-margin problem, support vectors can also lie inside the corridor or be misclassified.')
h('Part B — Hinge loss')
p('The elementwise hinge loss is max(0, 1 − margin). Adding Mai at (1, 2), labeled +1, gives score −1, margin −1 and hinge loss 2.')
output(12)
p('The original five points have no hinge penalty. Mai is the only point with positive loss for these fixed parameters; fitting a soft-margin model can change the set of active points.')
page()
h('Part C — A linear SVM from scratch')
p('The objective is mean(hinge) + (lambda / 2) × ||w||². Each iteration uses viol = margin < 1, grad_w = −X[viol].T @ y[viol] / N + lambda × w, and grad_b = −sum(y[viol]) / N. At margin exactly 1, zero is the chosen hinge subgradient. The intercept is not penalized.')
fig(16, 'Figure 1. Two overlapping synthetic classes; 200 observations per class.', 355)
p('Training uses lambda = 0.01, step size 0.05, zero initialization and 800 full-batch updates. Both terms of the recorded objective are evaluated after the update.')
output(17)
fig(17, 'Figure 2. Regularized hinge objective over the prescribed 800 updates.', 370)
page()
h('Part C — Solver comparison and convergence')
output(19)
p('For the averaged hinge objective, C = 1 / (lambda × N) = 0.25. LinearSVC also penalizes the intercept, so its coefficients are not an exact reference for the scratch objective. The additional linear SVC comparison leaves the intercept unpenalized.')
p('The scratch objective is 0.2011 versus 0.1682 for the objective-matched solver. Its second weight is still −0.3604, compared with approximately −0.0701 for SVC. The finite 800-step run is therefore not fully converged, even though its training accuracy is high. More iterations and a decaying step size can reduce optimization error.')
h('Part D — What the C dial does')
output(21)
p('As C increases, violations receive more weight relative to the norm penalty. In this run the corridor narrows and the support-vector count falls, then both plateau between C = 10 and 100. Training accuracy changes only slightly and is not monotonic. The count trend is an observation from this dataset, not a universal law.')
h('Part E — Synthetic Eedi-style attempts')
output(24)
p('The generating logit uses coefficients [1.8, −1.5, 1.0, 0.7] on accuracy, difficulty, topic match and practice; all other coefficients are zero. Bernoulli sampling adds outcome noise. The random split is 200 training, 100 validation and 100 test rows. The common-unit normal features are used as supplied; Part I isolates the effect of standardization.')
page()
h('Part F — L1 against L2')
p('Three logistic regressions are fitted on identical training rows: no penalty, L2 and L1. Penalized models use lambda = 0.02, C = 1 / (lambda × 200) = 0.25, solver liblinear and max_iter = 5000. Liblinear also penalizes its synthetic intercept coefficient.')
output(26)
fig(27, 'Figure 3. Coefficients without a penalty, with L2 and with L1. Dark bars are generating features.', 475)
p('L2 reduces the norm from 4.6055 to 2.3843 and keeps all coefficients nonzero. L1 reduces the norm to 2.3391 and sets 10 coefficients to numerical zero, defined by absolute value below 1e−8. It retains all four generating features and six noise features: sparsity does not guarantee exact feature recovery.')
p('Validation accuracy is 0.860 without regularization, 0.870 with L2 and 0.860 with L1. These are observations at one prescribed lambda, not an exhaustive comparison of the best possible L1 and L2 models.')
page()
h('Part G — Selecting lambda with validation')
p('The L2 search spans 24 log-spaced lambda values from 0.0001 to 10. Every model uses training rows only; validation accuracy chooses the grid point. A tie is resolved by taking the first maximum in ascending grid order, as specified by np.argmax.')
output(31)
fig(31, 'Figure 4. Training and validation accuracy across the lambda grid; dashed line marks the selected value.', 405)
p('The selected lambda is approximately 0.04062, with training accuracy 0.890 and validation accuracy 0.880. Training accuracy is a discrete score and need not decrease monotonically with regularization. Choosing by training accuracy would reward performance on data already used for fitting.')
h('Part H — Cross-validation and one test evaluation')
p('Five stratified folds are formed only from the 200 training rows. Each fold contains 160 fitting rows and 40 validation rows. C is recalculated with N_fold = 160, keeping the effective lambda fixed across folds. Using N = 200 inside every fold instead would make its penalty stronger than the label on the curve.')
output(35)
p('Cross-validation selects lambda approximately 0.1105 with mean accuracy 0.840. The chosen lambda is fixed before refitting on 300 training-plus-validation rows. The final model scores 0.790 on the 100 untouched test rows. This test result is not used to revise lambda or select another model.')
page()
h('Part I — Why standardization matters')
p('Multiplying feature 0 by 60 changes its units, not its information. A coefficient can then become much smaller while expressing a similar contribution to the score, making its L2 penalty cheaper. The unit change is also applied to validation rows before prediction.')
output(38)
output(39)
p('The fitted coefficient becomes about 45.7 times smaller and its squared-coefficient charge about 2,092 times smaller. With regularization, refitting does not produce an exact 60-fold inverse rescaling because the optimum changes. StandardScaler is fitted on training rows only. An assertion confirms that scaling the original-unit and changed-unit datasets produces matching validation probabilities.')
h('Part J — Responsible-use reflection')
p('A zero coefficient means the penalized fit did not retain a feature at this lambda on this sample; it does not establish that topic match is irrelevant for learning. Correlated predictors can be redundant, making L1’s retained feature list unstable across samples or penalty strengths. Even a useful feature can be removed by a strong penalty, so sparsity is not causal evidence.')
p('No: predicting an observed response is different from predicting the effect of giving extra practice. Motivation, prior ability and other confounders may influence both practice and success, so an association cannot identify that intervention’s effect. A randomized intervention, or a justified causal design with explicit assumptions, would be needed.')
p('The question’s dropped-topic-match premise is hypothetical. In the actual L1 run, feature index 2 is retained with coefficient approximately 0.78894.')
p('Limitations: these experiments use synthetic data and one held-out test split. The scratch SVM has remaining optimization error, and a short predictive feature list does not establish causal importance.', small)
page()
h('Appendix — Complete executed notebook code')
p('All code cells appear below in notebook order, including setup, self-checks, plotting and reflection. Line wrapping in this appendix is for display; the accompanying solved notebook preserves executable formatting.', small)
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    p(f'Notebook cell {i} · execution {c["execution_count"]}', heading)
    story.append(Preformatted(''.join(c['source']).rstrip(), code, maxLineLength=107))

prefix = cfg['group_prefix']
output_path = HERE / f'{prefix}_Lab5.pdf'
def decorate(canvas, doc):
    canvas.saveState()
    canvas.setFont('Arial', 8)
    canvas.setFillColor(navy)
    canvas.drawString(54, 810, 'MLA · Lab 5 · SVM and Regularization')
    canvas.setStrokeColor(colors.HexColor('#CBD5E1'))
    canvas.line(54, 801, 541, 801)
    canvas.line(54, 46, 541, 46)
    canvas.drawString(54, 33, prefix + ' · HANU · AY 2026–2027')
    canvas.drawRightString(541, 33, f'Page {doc.page}')
    canvas.restoreState()

SimpleDocTemplate(str(output_path), pagesize=A4, leftMargin=54, rightMargin=54,
                  topMargin=60, bottomMargin=60,
                  title='Lab 5 — SVM and Regularization',
                  author=cfg['submitting_member']['name']).build(
                      story, onFirstPage=decorate, onLaterPages=decorate)
(HERE / 'source_manifest.json').write_text(json.dumps({
    'source': str(SOURCE.relative_to(ROOT)),
    'sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'executed_code_cells': len(code_cells), 'figures': 4,
    'report': output_path.name,
    'structure': 'Notebook Parts A-J; previous reports used for visual style at user request',
}, indent=2) + '\n')
print(output_path)
