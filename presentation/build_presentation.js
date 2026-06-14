const pptxgen = require('pptxgenjs');
const html2pptx = require('/Users/chrism/.claude/plugins/marketplaces/claude-scientific-writer/skills/document-skills/pptx/scripts/html2pptx');
const fs = require('fs');
const path = require('path');

const SLIDES_DIR = path.join(__dirname, 'slides');
const FIGS_DIR = path.join(__dirname, 'figures');
const OUT = path.join(__dirname, '..', 'DAPIDL_Presentation_2026.pptx');

// Color constants (NO # prefix for pptxgenjs)
const C = {
  TEAL: '00526B',
  LTEAL: '0089A7',
  DARK: '003B4F',
  WHITE: 'FFFFFF',
  GOLD: 'E8A838',
  GREEN: '2ECC40',
  LGREEN: '7BC8A4',
  BLUE: '5DADE2',
  PURPLE: 'A569BD',
  GRAY: '666666',
  LGRAY: 'F5F7FA',
  RED: 'CC0000',
};

// Shared HTML template
function slide(title, body, opts = {}) {
  const bg = opts.bg || '#ffffff';
  const titleColor = opts.titleColor || '#00526B';
  const titleSize = opts.titleSize || '28pt';
  const padding = opts.padding || '40pt';
  return `<!DOCTYPE html><html><head><style>
html{background:${bg};}
body{width:720pt;height:405pt;margin:0;padding:0;font-family:Arial,sans-serif;display:flex;flex-direction:column;background:${bg};}
.wrap{margin:${padding};flex:1;display:flex;flex-direction:column;}
h1{color:${titleColor};font-size:${titleSize};margin:0 0 18pt 0;font-weight:bold;}
h2{color:#003B4F;font-size:20pt;margin:0 0 12pt 0;}
p,li{color:#333333;font-size:14pt;line-height:1.5;}
ul{margin:6pt 0;padding-left:24pt;}
li{margin-bottom:8pt;}
.two-col{display:flex;gap:30pt;flex:1;}
.col{flex:1;}
.accent{color:#0089A7;font-weight:bold;}
.small{font-size:11pt;color:#666;}
.highlight{background:#E8F4FD;border-left:4pt solid #00526B;padding:8pt 12pt;margin:10pt 0;}
</style></head><body><div class="wrap">
<h1>${title}</h1>
${body}
</div></body></html>`;
}

async function build() {
  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_16x9';
  pptx.author = 'Chris M';
  pptx.title = 'DAPIDL - Predicting Cell Types from DAPI';

  // ─── Slide 1: Title ───
  const s1html = `<!DOCTYPE html><html><head><style>
html{background:#ffffff;}
body{width:720pt;height:405pt;margin:0;padding:0;font-family:Arial,sans-serif;display:flex;flex-direction:column;justify-content:center;align-items:center;background:#ffffff;}
h1{color:#00526B;font-size:48pt;margin:0 0 12pt 0;font-weight:bold;text-align:center;}
p{color:#003B4F;font-size:18pt;text-align:center;margin:4pt 0;}
.sub{color:#666;font-size:13pt;margin-top:30pt;}
</style></head><body>
<h1>DAPIDL</h1>
<p>Predicting Cell Types from DAPI Nuclear Staining</p>
<p>Using Deep Learning &amp; Spatial Transcriptomics</p>
<p class="sub">March 2026</p>
</body></html>`;
  fs.writeFileSync(path.join(SLIDES_DIR, 's01.html'), s1html);
  await html2pptx(path.join(SLIDES_DIR, 's01.html'), pptx);

  // ─── Slide 2: Goal ───
  fs.writeFileSync(path.join(SLIDES_DIR, 's02.html'), slide(
    'Goal',
    `<div class="highlight"><p><b>Automatic cell type annotation using Deep Learning on DAPI-stained tissue sections</b></p></div>
    <ul>
      <li>Train a CNN to predict cell types from <span class="accent">nuclear morphology alone</span></li>
      <li>Use spatial transcriptomics (Xenium, MERSCOPE) as <span class="accent">automatic training data generator</span></li>
      <li>Eliminate the need for expensive gene expression assays at inference time</li>
      <li>Enable retrospective analysis of existing DAPI-stained tissue archives</li>
    </ul>`
  ));
  await html2pptx(path.join(SLIDES_DIR, 's02.html'), pptx);

  // ─── Slide 3: Gold Standard vs DAPIDL ───
  const s3html = `<!DOCTYPE html><html><head><style>
html{background:#fff;}
body{width:720pt;height:405pt;margin:0;padding:0;font-family:Arial,sans-serif;display:flex;flex-direction:column;background:#fff;}
.wrap{margin:40pt;flex:1;}
h1{color:#00526B;font-size:28pt;margin:0 0 24pt 0;}
.compare{display:flex;gap:40pt;justify-content:center;align-items:flex-start;}
.side{flex:1;text-align:center;}
h2{font-size:22pt;margin:0 0 16pt 0;}
.left h2{color:#CC0000;}
.right h2{color:#0089A7;}
.row{display:flex;align-items:center;gap:12pt;margin-bottom:16pt;justify-content:center;}
p{font-size:15pt;color:#003B4F;margin:0;}
.icon-space{width:50pt;height:50pt;}
</style></head><body><div class="wrap">
<h1>Current Gold Standard vs. DAPIDL</h1>
<div class="compare">
  <div class="side left">
    <h2>Manual Pathology</h2>
    <div class="row"><p>Subjective assessment</p></div>
    <div class="row"><p>Slow (hours per slide)</p></div>
    <div class="row"><p>High cost per annotation</p></div>
    <div class="row"><p>Expert-dependent</p></div>
  </div>
  <div class="side right">
    <h2>DAPIDL</h2>
    <div class="row"><p>Fully automated &amp; objective</p></div>
    <div class="row"><p>Fast (&lt;1s / 1000 cells)</p></div>
    <div class="row"><p>Low marginal cost</p></div>
    <div class="row"><p>Reproducible &amp; scalable</p></div>
  </div>
</div>
</div></body></html>`;
  fs.writeFileSync(path.join(SLIDES_DIR, 's03.html'), s3html);
  await html2pptx(path.join(SLIDES_DIR, 's03.html'), pptx);

  // ─── Slide 4: Spatial Transcriptomics ───
  fs.writeFileSync(path.join(SLIDES_DIR, 's04.html'), slide(
    'Spatial Transcriptomics',
    `<ul>
      <li>Technologies that measure <span class="accent">gene expression at single-cell resolution</span> while preserving spatial context</li>
      <li>Platforms: <b>Xenium</b> (10x Genomics), <b>MERSCOPE</b> (Vizgen), CosMx (NanoString)</li>
      <li>Output per cell: transcript counts for 300-5000 genes + spatial coordinates</li>
      <li>All platforms include <span class="accent">DAPI nuclear staining</span> as morphology channel</li>
      <li>Enable understanding of intra-tumor heterogeneity and the tumor microenvironment</li>
    </ul>
    <div class="highlight">
      <p><b>Key insight:</b> DAPI staining is universal across all spatial transcriptomics platforms. A model that learns from one dataset can predict on any DAPI image.</p>
    </div>`
  ));
  await html2pptx(path.join(SLIDES_DIR, 's04.html'), pptx);

  // ─── Slide 5: How do they work ───
  fs.writeFileSync(path.join(SLIDES_DIR, 's05.html'), slide(
    'How Spatial Transcriptomics Works',
    `<ul>
      <li>Tissue section placed on slides with <span class="accent">prepared probe spots</span></li>
      <li>Probes bind complementary mRNA molecules in situ (FISH-based)</li>
      <li>Fluorescent readout identifies which transcripts are present in each cell</li>
      <li>Cell segmentation assigns transcripts to individual cells</li>
      <li>Result: <b>gene expression matrix + spatial coordinates + DAPI image</b></li>
    </ul>
    <div class="highlight">
      <p><b>Limitation:</b> Works for 300-5000 pre-selected genes (panel-based). Not whole-genome.</p>
    </div>`
  ));
  await html2pptx(path.join(SLIDES_DIR, 's05.html'), pptx);

  // ─── Slide 6: The Challenge ───
  fs.writeFileSync(path.join(SLIDES_DIR, 's06.html'), slide(
    'The Challenge: Training Data Quality',
    `<ul>
      <li>Training the CNN is <span class="accent">straightforward</span> once labels exist</li>
      <li>Creating sufficient <b>high-quality training labels</b> is the real challenge</li>
      <li>No pathologist-annotated ground truth available for most datasets</li>
      <li>Automated annotation methods each have biases and error rates</li>
      <li>Label noise directly degrades model convergence and accuracy</li>
    </ul>
    <div class="highlight">
      <p><b>Our solution:</b> Democratic consensus of 11 independent annotation algorithms. Only high-confidence cells enter the training set.</p>
    </div>`
  ));
  await html2pptx(path.join(SLIDES_DIR, 's06.html'), pptx);

  // ─── Slide 7: DAPIDL Pipeline (with image) ───
  const s7html = `<!DOCTYPE html><html><head><style>
html{background:#fff;}
body{width:720pt;height:405pt;margin:0;padding:0;font-family:Arial,sans-serif;display:flex;flex-direction:column;background:#fff;}
.wrap{margin:30pt 40pt;flex:1;display:flex;flex-direction:column;}
h1{color:#00526B;font-size:28pt;margin:0 0 10pt 0;}
.content{display:flex;gap:20pt;flex:1;align-items:flex-start;}
.img-col{flex:0 0 260pt;}
.text-col{flex:1;}
p{font-size:12pt;color:#333;margin:4pt 0;}
.step{margin-bottom:6pt;}
.step-num{color:#00526B;font-weight:bold;font-size:13pt;}
</style></head><body><div class="wrap">
<h1>DAPIDL Pipeline</h1>
<div class="content">
  <div class="img-col">
    <img src="${path.join(FIGS_DIR, 'pipeline_flow.png')}" style="width:250pt;height:auto;">
  </div>
  <div class="text-col">
    <div class="step"><p><span class="step-num">1. Input:</span> Raw spatial transcriptomics data (Xenium, MERSCOPE, STHELAR)</p></div>
    <div class="step"><p><span class="step-num">2. Annotate:</span> 11 methods vote on each cell's type using gene expression</p></div>
    <div class="step"><p><span class="step-num">3. Standardize:</span> Map labels to Cell Ontology for cross-dataset consistency</p></div>
    <div class="step"><p><span class="step-num">4. Filter:</span> Remove low-confidence cells (marker enrichment + spatial coherence)</p></div>
    <div class="step"><p><span class="step-num">5. Extract:</span> Cut nucleus-centered DAPI patches, save as LMDB</p></div>
    <div class="step"><p><span class="step-num">6. Train:</span> CNN/ViT learns to predict cell type from DAPI morphology</p></div>
    <div class="step"><p><span class="step-num">7. Output:</span> Deployable model predicts on any DAPI image</p></div>
    <p style="margin-top:10pt;font-size:11pt;color:#666;">Fully modular, configurable, and orchestrated via ClearML pipelines.</p>
  </div>
</div>
</div></body></html>`;
  fs.writeFileSync(path.join(SLIDES_DIR, 's07.html'), s7html);
  await html2pptx(path.join(SLIDES_DIR, 's07.html'), pptx);

  // ─── Slide 8: Ensemble Annotation ───
  const s8html = `<!DOCTYPE html><html><head><style>
html{background:#fff;}
body{width:720pt;height:405pt;margin:0;padding:0;font-family:Arial,sans-serif;display:flex;flex-direction:column;background:#fff;}
.wrap{margin:30pt 40pt;flex:1;display:flex;flex-direction:column;}
h1{color:#00526B;font-size:26pt;margin:0 0 10pt 0;}
.content{display:flex;gap:20pt;flex:1;}
.img-col{flex:0 0 260pt;display:flex;align-items:center;}
.text-col{flex:1;}
p{font-size:13pt;color:#333;margin:5pt 0;}
ul{padding-left:18pt;margin:4pt 0;}
li{font-size:12pt;color:#333;margin-bottom:4pt;}
.accent{color:#0089A7;font-weight:bold;}
</style></head><body><div class="wrap">
<h1>Automatic Annotation via Algorithmic Consensus</h1>
<div class="content">
  <div class="img-col">
    <img src="${path.join(FIGS_DIR, 'ensemble_funnel.png')}" style="width:250pt;height:auto;">
  </div>
  <div class="text-col">
    <p><b>What they use:</b></p>
    <ul>
      <li>Gene expression transcripts</li>
      <li>Cell marker databases (CellMarker 2.0, PanglaoDB)</li>
      <li>Reference atlases (Tabula Sapiens, Human Cell Atlas)</li>
    </ul>
    <p><span class="accent">Democratized ground truth:</span> We rely on majority vote of 11 distinct algorithms to generate training labels without manual annotation.</p>
    <p style="font-size:11pt;color:#666;margin-top:8pt;">Methods: CellTypist, SingleR, ScType, SCINA, PopV, Azimuth, ScANVI, ScArches, Clustifyr, Garnett, ML classifiers</p>
  </div>
</div>
</div></body></html>`;
  fs.writeFileSync(path.join(SLIDES_DIR, 's08.html'), s8html);
  await html2pptx(path.join(SLIDES_DIR, 's08.html'), pptx);

  // ─── Slide 9: Ensemble Voting Details ───
  fs.writeFileSync(path.join(SLIDES_DIR, 's09.html'), slide(
    'Ensemble Voting System',
    `<div class="two-col">
      <div class="col">
        <h2>How It Works</h2>
        <ul>
          <li>Each method independently labels every cell</li>
          <li>Per-cell majority vote determines final label</li>
          <li><span class="accent">Unweighted voting outperforms</span> confidence-weighted by 15-22%</li>
          <li>Agreement rate determines confidence tier</li>
        </ul>
      </div>
      <div class="col">
        <h2>Why This Works</h2>
        <ul>
          <li>Different methods use different algorithms and references</li>
          <li>Errors are uncorrelated across methods</li>
          <li>Consensus eliminates individual method biases</li>
          <li>Achieves <span class="accent">~88% accuracy, 0.84 macro F1</span> on breast cancer (vs. ground truth)</li>
        </ul>
      </div>
    </div>`,
    { titleSize: '26pt' }
  ));
  await html2pptx(path.join(SLIDES_DIR, 's09.html'), pptx);

  // ─── Slide 10: Confidence Tiers ───
  const s10html = `<!DOCTYPE html><html><head><style>
html{background:#fff;}
body{width:720pt;height:405pt;margin:0;padding:0;font-family:Arial,sans-serif;display:flex;flex-direction:column;background:#fff;}
.wrap{margin:30pt 40pt;flex:1;display:flex;flex-direction:column;}
h1{color:#00526B;font-size:26pt;margin:0 0 10pt 0;}
.content{display:flex;gap:30pt;flex:1;align-items:center;}
.img-col{flex:0 0 260pt;}
.text-col{flex:1;}
p{font-size:13pt;color:#333;margin:6pt 0;}
.accent{color:#0089A7;font-weight:bold;}
</style></head><body><div class="wrap">
<h1>Quality Control via Confidence Tiers</h1>
<div class="content">
  <div class="img-col">
    <img src="${path.join(FIGS_DIR, 'confidence_tiers.png')}" style="width:250pt;height:auto;">
  </div>
  <div class="text-col">
    <p><b>Three-pass hierarchical filtering:</b></p>
    <p><span class="accent">Tier 1:</span> All methods agree (100%). Highest training priority.</p>
    <p><span class="accent">Tier 2:</span> >80% agreement. Good quality, included with lower weight.</p>
    <p><span class="accent">Tier 3:</span> >50% agreement. Used cautiously for rare cell types.</p>
    <p style="color:#CC0000;"><b>Discarded:</b> &lt;50% agreement. Ambiguous cells excluded to prevent confusing the model.</p>
    <p style="font-size:11pt;color:#666;margin-top:8pt;">Additional quality signals: marker gene enrichment, spatial coherence (KNN), tissue-specific proportion plausibility.</p>
  </div>
</div>
</div></body></html>`;
  fs.writeFileSync(path.join(SLIDES_DIR, 's10.html'), s10html);
  await html2pptx(path.join(SLIDES_DIR, 's10.html'), pptx);

  // ─── Slide 11: Cell Ontology ───
  const s11html = `<!DOCTYPE html><html><head><style>
html{background:#fff;}
body{width:720pt;height:405pt;margin:0;padding:0;font-family:Arial,sans-serif;display:flex;flex-direction:column;background:#fff;}
.wrap{margin:30pt 40pt;flex:1;display:flex;flex-direction:column;}
h1{color:#00526B;font-size:24pt;margin:0 0 10pt 0;}
.content{display:flex;gap:20pt;flex:1;align-items:flex-start;}
.img-col{flex:0 0 280pt;display:flex;align-items:center;}
.text-col{flex:1;}
p{font-size:13pt;color:#333;margin:6pt 0;}
ul{padding-left:18pt;}
li{font-size:12pt;color:#333;margin-bottom:5pt;}
.accent{color:#0089A7;font-weight:bold;}
</style></head><body><div class="wrap">
<h1>Standardized Language: Cell Ontology (CL)</h1>
<div class="content">
  <div class="img-col">
    <img src="${path.join(FIGS_DIR, 'hierarchy_tree.png')}" style="width:270pt;height:auto;">
  </div>
  <div class="text-col">
    <p><b>Problem:</b> Each annotation method uses different names for the same cell type.</p>
    <ul>
      <li>"T_CD8", "Cytotoxic T", "CD8+ T-Cell", "T-Lymphocyte (CD8)" all mean the same thing</li>
    </ul>
    <p><b>Solution:</b> CL Mapper standardizes all labels to the Cell Ontology (OBO).</p>
    <ul>
      <li><span class="accent">Exact match</span> → Fuzzy match (>0.85) → Alias lookup</li>
      <li>Hierarchical: Broad (3) → Medium (~15) → Fine (~50 classes)</li>
      <li>Fine-grained labels only when confidence is high</li>
    </ul>
  </div>
</div>
</div></body></html>`;
  fs.writeFileSync(path.join(SLIDES_DIR, 's11.html'), s11html);
  await html2pptx(path.join(SLIDES_DIR, 's11.html'), pptx);

  // ─── Slide 12: How Backbones Work ───
  fs.writeFileSync(path.join(SLIDES_DIR, 's12.html'), slide(
    'How Backbones Work',
    `<ul>
      <li><b>Input:</b> Single-nucleus DAPI image (128x128 px, 1 channel)</li>
      <li><b>Channel Adaptation:</b> Replicate to 3 channels for pretrained RGB networks</li>
      <li><b>Backbone:</b> EfficientNetV2-S extracts a 1792-dim feature vector</li>
      <li><b>Classification heads:</b> Three heads for Broad (3), Medium (~15), Fine (~50) classes</li>
    </ul>
    <div class="highlight">
      <p><b>Transfer learning:</b> Networks pretrained on ImageNet already recognize edges, textures, and shapes. These low-level features transfer directly to nuclear morphology.</p>
    </div>
    <p class="small">Available backbones: EfficientNetV2-S (default, 20M params), ResNet-18/34, ConvNeXt-Tiny/Small, ViT-Base/Large, Phikon-v2 (pathology), UNI (pathology). LoRA fine-tuning supported via PEFT.</p>`
  ));
  await html2pptx(path.join(SLIDES_DIR, 's12.html'), pptx);

  // ─── Slide 12b: CNN Architecture Diagram ───
  const s12b = `<!DOCTYPE html><html><head><style>
html{background:#fff;}
body{width:720pt;height:405pt;margin:0;padding:0;font-family:Arial,sans-serif;display:flex;flex-direction:column;background:#fff;}
.wrap{margin:30pt 30pt;flex:1;display:flex;flex-direction:column;}
h1{color:#00526B;font-size:26pt;margin:0 0 8pt 0;}
p{font-size:12pt;color:#333;margin:4pt 0;}
.img-wrap{flex:1;display:flex;align-items:center;justify-content:center;}
.small{font-size:10pt;color:#666;}
</style></head><body><div class="wrap">
<h1>CNN Architecture (EfficientNetV2-S)</h1>
<div class="img-wrap">
  <img src="${path.join(FIGS_DIR, 'cnn_architecture.png')}" style="width:650pt;height:auto;">
</div>
<p class="small">Stacked convolutional blocks progressively reduce spatial dimensions while increasing feature depth. Global average pooling produces a fixed-size feature vector for the classification heads.</p>
</div></body></html>`;
  fs.writeFileSync(path.join(SLIDES_DIR, 's12b.html'), s12b);
  await html2pptx(path.join(SLIDES_DIR, 's12b.html'), pptx);

  // ─── Slide 12c: ViT Architecture Diagram ───
  const s12c = `<!DOCTYPE html><html><head><style>
html{background:#fff;}
body{width:720pt;height:405pt;margin:0;padding:0;font-family:Arial,sans-serif;display:flex;flex-direction:column;background:#fff;}
.wrap{margin:30pt 30pt;flex:1;display:flex;flex-direction:column;}
h1{color:#00526B;font-size:26pt;margin:0 0 8pt 0;}
p{font-size:12pt;color:#333;margin:4pt 0;}
.img-wrap{flex:1;display:flex;align-items:center;justify-content:center;}
.small{font-size:10pt;color:#666;}
</style></head><body><div class="wrap">
<h1>Vision Transformer (ViT) Architecture</h1>
<div class="img-wrap">
  <img src="${path.join(FIGS_DIR, 'vit_architecture.png')}" style="width:650pt;height:auto;">
</div>
<p class="small">The image is split into fixed-size patches, linearly projected into embeddings, and processed by a stack of Transformer encoder blocks using multi-head self-attention. The [CLS] token aggregates global information for classification.</p>
</div></body></html>`;
  fs.writeFileSync(path.join(SLIDES_DIR, 's12c.html'), s12c);
  await html2pptx(path.join(SLIDES_DIR, 's12c.html'), pptx);

  // ─── Slide 13: Transfer Learning ───
  fs.writeFileSync(path.join(SLIDES_DIR, 's13.html'), slide(
    'Transfer Learning: Why It Works',
    `<div class="two-col">
      <div class="col">
        <h2>Pre-training (ImageNet)</h2>
        <ul>
          <li>Trained on 14M+ labeled images</li>
          <li>Learns universal visual features: edges, curves, gradients, textures</li>
          <li>These features are domain-agnostic</li>
        </ul>
      </div>
      <div class="col">
        <h2>Fine-tuning (DAPI)</h2>
        <ul>
          <li>The same edges and textures appear in nuclei</li>
          <li>Only needs to learn domain-specific patterns</li>
          <li>Faster convergence, better generalization</li>
        </ul>
      </div>
    </div>
    <div class="highlight">
      <p>A model that can recognize edges of a wolf already knows how to recognize edges of a cell. We skip expensive base training and only fine-tune the classification head.</p>
    </div>`
  ));
  await html2pptx(path.join(SLIDES_DIR, 's13.html'), pptx);

  // ─── Slide 14: Curriculum Learning ───
  fs.writeFileSync(path.join(SLIDES_DIR, 's14.html'), slide(
    'Curriculum Learning: Coarse to Fine',
    `<div class="two-col">
      <div class="col">
        <h2>Hierarchical Training Strategy</h2>
        <ul>
          <li><span class="accent">Phase 1 (Broad):</span> Epithelial vs. Immune vs. Stromal</li>
          <li><span class="accent">Phase 2 (Medium):</span> T-Cell, B-Cell, Macrophage, Luminal, Fibroblast...</li>
          <li><span class="accent">Phase 3 (Fine):</span> CD4+ T, CD8+ T, M1 Macrophage...</li>
        </ul>
        <p style="font-size:12pt;color:#666;margin-top:10pt;">Each phase progressively activates finer classification heads while maintaining coarse accuracy.</p>
      </div>
      <div class="col">
        <h2>Confidence-Based Fallback</h2>
        <ul>
          <li>At inference: try fine prediction first</li>
          <li>If confidence &lt; threshold, fall back to medium</li>
          <li>If still uncertain, report at broad level</li>
          <li>Result: <b>every cell gets a reliable label</b></li>
        </ul>
      </div>
    </div>`
  ));
  await html2pptx(path.join(SLIDES_DIR, 's14.html'), pptx);

  // ─── Slide 15: Training Loop ───
  const s15html = `<!DOCTYPE html><html><head><style>
html{background:#fff;}
body{width:720pt;height:405pt;margin:0;padding:0;font-family:Arial,sans-serif;display:flex;flex-direction:column;background:#fff;}
.wrap{margin:30pt 40pt;flex:1;display:flex;flex-direction:column;}
h1{color:#00526B;font-size:26pt;margin:0 0 10pt 0;}
.content{display:flex;gap:20pt;flex:1;align-items:center;}
.img-col{flex:0 0 280pt;}
.text-col{flex:1;}
p{font-size:13pt;color:#333;margin:6pt 0;}
.accent{color:#0089A7;font-weight:bold;}
</style></head><body><div class="wrap">
<h1>Training Loop &amp; Validation Strategy</h1>
<div class="content">
  <div class="img-col">
    <img src="${path.join(FIGS_DIR, 'training_curve.png')}" style="width:270pt;height:auto;">
  </div>
  <div class="text-col">
    <p><b>Curriculum Phase Transitions:</b></p>
    <p>When switching from broad to medium, the new classification head unfreezes. This causes a <span class="accent">temporary accuracy dip</span> and loss spike as the model adapts to finer categories.</p>
    <p>The same dip recurs at the medium→fine transition. Each time, the model recovers and surpasses its prior level.</p>
    <p><b>Why the final plateau is lower:</b></p>
    <p>Fine-grained cell subtypes (CD4+ vs CD8+) are harder to distinguish from DAPI morphology alone. The model converges at a lower ceiling — this is a biological, not a technical, limit.</p>
    <p style="font-size:10pt;color:#666;">Early stopping (patience=15), AdamW, cosine annealing, MixUp (0.2), DALI GPU loading.</p>
  </div>
</div>
</div></body></html>`;
  fs.writeFileSync(path.join(SLIDES_DIR, 's15.html'), s15html);
  await html2pptx(path.join(SLIDES_DIR, 's15.html'), pptx);

  // ─── Slide 16: Patch Sizes ───
  fs.writeFileSync(path.join(SLIDES_DIR, 's16.html'), slide(
    'Patch Size Comparison',
    `<div class="two-col">
      <div class="col">
        <h2>Nucleus-Centered Patches</h2>
        <ul>
          <li><b>32x32:</b> Only nucleus interior. Misses surrounding context.</li>
          <li><b>64x64:</b> Nucleus + immediate neighborhood.</li>
          <li><b>128x128:</b> Good balance of nucleus detail + local tissue context.</li>
          <li><b>256x256:</b> Includes multiple neighbors. <span class="accent">Best F1 (+15% over 128)</span></li>
        </ul>
      </div>
      <div class="col">
        <h2>Key Finding</h2>
        <p><span class="accent">Larger patches = better performance.</span></p>
        <p>Local tissue context (neighboring cells, tissue architecture) provides critical classification cues beyond nuclear morphology.</p>
        <p>Trade-off: larger patches need more GPU memory and slower training.</p>
        <p style="font-size:11pt;color:#666;margin-top:8pt;">Physical-space normalization ensures consistent scale across platforms (target: 0.2125 um/px).</p>
      </div>
    </div>`
  ));
  await html2pptx(path.join(SLIDES_DIR, 's16.html'), pptx);

  // ─── Slide 17: Results ───
  const s17html = `<!DOCTYPE html><html><head><style>
html{background:#fff;}
body{width:720pt;height:405pt;margin:0;padding:0;font-family:Arial,sans-serif;display:flex;flex-direction:column;background:#fff;}
.wrap{margin:30pt 40pt;flex:1;display:flex;flex-direction:column;}
h1{color:#00526B;font-size:26pt;margin:0 0 12pt 0;}
p{font-size:13pt;color:#333;margin:4pt 0;}
.placeholder{width:320pt;height:180pt;background:#F0F4F8;margin:0 auto;}
.two-col{display:flex;gap:20pt;flex:1;}
.col{flex:1;}
h2{color:#003B4F;font-size:18pt;margin:0 0 8pt 0;}
.metric{margin-bottom:6pt;}
.val{color:#0089A7;font-size:20pt;font-weight:bold;}
.label{font-size:11pt;color:#666;}
</style></head><body><div class="wrap">
<h1>Results: Performance Metrics</h1>
<div class="two-col">
  <div class="col">
    <h2>Broad Classification (3 classes)</h2>
    <div class="metric"><p><span class="val">0.90</span></p><p class="label">Macro F1 (Consensus Labels)</p></div>
    <div class="metric"><p><span class="val">89%</span></p><p class="label">Overall Accuracy</p></div>
    <div class="metric"><p><span class="val">0.82</span></p><p class="label">Macro F1 (Ground Truth Labels)</p></div>
  </div>
  <div class="col">
    <h2>Fine-Grained (17 classes)</h2>
    <div class="metric"><p><span class="val">0.41</span></p><p class="label">Macro F1 (best EfficientNetV2-S)</p></div>
    <div class="metric"><p><span class="val">67%</span></p><p class="label">Overall Accuracy</p></div>
    <p style="font-size:12pt;color:#666;margin-top:8pt;"><b>Per-class highlights:</b> Epithelial Luminal 0.92, T-Cell 0.80, B-Cell 0.77. Stromal subtypes remain challenging (Fibroblast 0.36).</p>
  </div>
</div>
<p style="font-size:10pt;color:#999;margin-top:6pt;">Xenium breast cancer dataset, 167,780 cells. EfficientNetV2-S backbone, 128px patches, consensus annotation.</p>
</div></body></html>`;
  fs.writeFileSync(path.join(SLIDES_DIR, 's17.html'), s17html);
  const { slide: s17slide, placeholders: p17 } = await html2pptx(path.join(SLIDES_DIR, 's17.html'), pptx);

  // ─── Slide 18: Cross-Platform ───
  fs.writeFileSync(path.join(SLIDES_DIR, 's18.html'), slide(
    'Cross-Platform Compatibility',
    `<div class="two-col">
      <div class="col">
        <h2>Why DAPIDL is Platform-Agnostic</h2>
        <ul>
          <li>DAPI staining chemistry is <span class="accent">identical</span> across Xenium, MERSCOPE, CosMx</li>
          <li>Gene expression only used for training labels (not at inference)</li>
          <li>Adaptive normalization handles intensity differences (~16x between platforms)</li>
        </ul>
      </div>
      <div class="col">
        <h2>Supported Platforms</h2>
        <ul>
          <li><b>Xenium</b> (10x Genomics): 341 genes, 0.2125 um/px</li>
          <li><b>MERSCOPE</b> (Vizgen): 500 genes, 0.108 um/px</li>
          <li><b>STHELAR</b>: 31 sections, 16 tissue types, co-registered DAPI+H&amp;E</li>
          <li><b>Any DAPI image</b> at inference (no transcriptomics needed)</li>
        </ul>
      </div>
    </div>`
  ));
  await html2pptx(path.join(SLIDES_DIR, 's18.html'), pptx);

  // ─── Slide 19: Foundation Models ───
  fs.writeFileSync(path.join(SLIDES_DIR, 's19.html'), slide(
    'Foundation Models &amp; LoRA Fine-Tuning',
    `<div class="two-col">
      <div class="col">
        <h2>Pathology Foundation Models</h2>
        <ul>
          <li><b>Phikon-v2</b> (Owkin): ViT-L/16, trained on 100M+ pathology patches</li>
          <li><b>UNI</b> (Mahmood Lab): ViT-L/16, pan-cancer histopathology</li>
          <li>Native 3-channel input (H&amp;E staining)</li>
        </ul>
        <p style="font-size:12pt;color:#666;">Finding: ImageNet-pretrained EfficientNetV2-S matches Phikon-v2 on DAPI (F1=0.41 vs 0.41). DAPI differs enough from H&amp;E that pathology pretraining provides no advantage.</p>
      </div>
      <div class="col">
        <h2>LoRA Fine-Tuning (PEFT)</h2>
        <ul>
          <li>Low-Rank Adaptation: freeze backbone, train small adapter matrices</li>
          <li>Reduces trainable parameters by <span class="accent">~95%</span></li>
          <li>Faster training, lower GPU memory</li>
          <li>Enables efficient adaptation to new tissue types</li>
        </ul>
      </div>
    </div>`,
    { titleSize: '24pt' }
  ));
  await html2pptx(path.join(SLIDES_DIR, 's19.html'), pptx);

  // ─── Slide 20: Validation Without GT ───
  fs.writeFileSync(path.join(SLIDES_DIR, 's20.html'), slide(
    'Validation Without Ground Truth',
    `<p><b>Challenge:</b> Most spatial transcriptomics datasets lack pathologist-annotated ground truth.</p>
    <div class="two-col">
      <div class="col">
        <h2>Cross-Modal Validation</h2>
        <ul>
          <li><b>Leiden clustering:</b> Compare unsupervised clusters with supervised labels (ARI, NMI)</li>
          <li><b>DAPI morphology:</b> Use trained CNN as independent validation (agreement rate)</li>
          <li><b>Multi-method consensus:</b> Agreement between annotation methods</li>
        </ul>
      </div>
      <div class="col">
        <h2>Marker Gene Validation</h2>
        <ul>
          <li>Unified marker database (Cell Marker Accordion + CellMarker 2.0)</li>
          <li>Verify that predicted cell types express expected markers</li>
          <li>Spatial coherence: cells of same type should cluster spatially</li>
          <li>Proportion plausibility: tissue-specific expected ratios</li>
        </ul>
      </div>
    </div>`,
    { titleSize: '24pt' }
  ));
  await html2pptx(path.join(SLIDES_DIR, 's20.html'), pptx);

  // ─── Slide 21: popV ───
  fs.writeFileSync(path.join(SLIDES_DIR, 's21.html'), slide(
    'popV: Popular Vote Annotation',
    `<ul>
      <li>Framework combining <span class="accent">multiple cell-type transfer tools</span> for consensus classification (Ergen et al., Nature Genetics 2024)</li>
      <li>Uses Tabula Sapiens as reference atlas with organ-specific subsets</li>
      <li>Consensus voting reduces bias of individual tools</li>
      <li>Predicts cell types with stated confidence per cell</li>
      <li>Generates labels that are highly likely to match reference atlas cell types</li>
    </ul>
    <div class="highlight">
      <p>popV achieves <span class="accent">0.90 F1</span> on Xenium breast cancer when combined with our confidence filtering, matching the best single-method performance while being more robust.</p>
    </div>`
  ));
  await html2pptx(path.join(SLIDES_DIR, 's21.html'), pptx);

  // ─── Slide 22: CellTypist ───
  fs.writeFileSync(path.join(SLIDES_DIR, 's22.html'), slide(
    'CellTypist: Machine Learning Annotation',
    `<ul>
      <li>Logistic regression-based classifier trained on the <span class="accent">Human Cell Atlas</span></li>
      <li>Model zoo: 100+ pretrained models for different tissues and cell types</li>
      <li>Input: normalized gene expression vector per cell</li>
      <li>Output: cell type label + probability score</li>
      <li>Key models for DAPIDL: Cells_Adult_Breast.pkl, Immune_All_High.pkl, Immune_All_Low.pkl</li>
    </ul>
    <div class="highlight">
      <p><b>Multi-model strategy:</b> Running 3-5 CellTypist models in parallel and combining predictions achieves better coverage than any single model.</p>
    </div>
    <p class="small">Dominguez Conde et al., Science 2022.</p>`
  ));
  await html2pptx(path.join(SLIDES_DIR, 's22.html'), pptx);

  // ─── Slide 23: Hyperparameters ───
  fs.writeFileSync(path.join(SLIDES_DIR, 's23.html'), slide(
    'Configurable Parameters',
    `<div class="two-col">
      <div class="col">
        <h2>Data Parameters</h2>
        <ul>
          <li><b>Patch size:</b> 32, 64, 128, 256 px</li>
          <li><b>Normalization:</b> adaptive, percentile, minmax</li>
          <li><b>Confidence tier:</b> 1, 2, or 3</li>
          <li><b>Annotation methods:</b> any subset of 11</li>
          <li><b>Granularity:</b> broad / coarse / fine</li>
          <li><b>Segmentation:</b> Xenium native or Cellpose</li>
        </ul>
      </div>
      <div class="col">
        <h2>Training Parameters</h2>
        <ul>
          <li><b>Backbone:</b> EfficientNet, ResNet, ConvNeXt, ViT, Phikon-v2</li>
          <li><b>Learning rate:</b> 1e-4 (cosine annealing)</li>
          <li><b>Batch size:</b> 64-256</li>
          <li><b>Augmentation:</b> MixUp, rotation, flip, noise</li>
          <li><b>Class weighting:</b> inverse frequency with cap (10x)</li>
          <li><b>HPO:</b> Optuna-based Bayesian optimization via ClearML</li>
        </ul>
      </div>
    </div>`
  ));
  await html2pptx(path.join(SLIDES_DIR, 's23.html'), pptx);

  // ─── Slide 24: Advantages ───
  fs.writeFileSync(path.join(SLIDES_DIR, 's24.html'), slide(
    'Advantages &amp; Clinical Impact',
    `<ul>
      <li>DAPI staining is <span class="accent">already performed routinely</span> in most histology labs</li>
      <li>Cell types can be determined <b>years after tissue collection</b> from archived slides</li>
      <li>No expensive spatial transcriptomics equipment needed at inference</li>
      <li><span class="accent">Platform-agnostic:</span> train on Xenium, predict on any DAPI image</li>
      <li>Fully reproducible and scalable (no human subjectivity)</li>
      <li>Modular pipeline: easily add new tissues, methods, or backbones</li>
    </ul>
    <div class="highlight">
      <p><b>Vision:</b> A universal DAPI classifier that works across tissue types, enabling rapid cell type mapping in any pathology lab with just standard nuclear staining.</p>
    </div>`
  ));
  await html2pptx(path.join(SLIDES_DIR, 's24.html'), pptx);

  // ─── Slide 25: Key References ───
  fs.writeFileSync(path.join(SLIDES_DIR, 's25.html'), slide(
    'Key References',
    `<ul>
      <li>Ergen et al. (2024) "Consensus prediction of cell type labels with popV" <i>Nat Genetics</i></li>
      <li>Aran et al. (2019) "SingleR: Reference-based single-cell annotation" <i>Nat Immunology</i></li>
      <li>Biancalani et al. (2021) "Tangram: Spatial alignment of single-cell transcriptomes" <i>Nat Methods</i></li>
      <li>Osumi-Sutherland et al. (2021) "Cell type ontologies of the Human Cell Atlas" <i>Nat Cell Bio</i></li>
      <li>Hua et al. (2026) "NuSPIRe: Self-supervised pretraining for nuclear morphology" <i>Genome Biology</i></li>
      <li>Diehl et al. (2016) "The Cell Ontology 2016: Enhanced content and interoperability" <i>J Biomed Semantics</i></li>
      <li>Wang et al. (2021) "Leveraging Cell Ontology to classify unseen cell types" <i>Nat Comms</i></li>
    </ul>
    <p class="small">Full bibliography in Zotero collection "dapiDL".</p>`,
    { titleSize: '24pt', padding: '30pt' }
  ));
  await html2pptx(path.join(SLIDES_DIR, 's25.html'), pptx);

  // ─── Slide 26: Summary ───
  const s26html = `<!DOCTYPE html><html><head><style>
html{background:#00526B;}
body{width:720pt;height:405pt;margin:0;padding:0;font-family:Arial,sans-serif;display:flex;flex-direction:column;justify-content:center;align-items:center;background:#00526B;}
h1{color:#ffffff;font-size:36pt;margin:0 0 20pt 0;text-align:center;}
p{color:#B3D9F2;font-size:16pt;text-align:center;margin:6pt 40pt;}
.divider{width:80pt;height:3pt;background:#E8A838;margin:20pt auto;}
</style></head><body>
<h1>Summary &amp; Outlook</h1>
<div class="divider" style="width:80pt;height:3pt;background:#E8A838;"></div>
<p>DAPIDL demonstrates that cell type prediction from DAPI nuclear staining alone is feasible, achieving 0.90 F1 at broad level and 0.41 F1 for 17 fine-grained classes.</p>
<p>The modular ClearML pipeline supports 3 platforms, 11 annotation methods, and multiple backbone architectures.</p>
<p style="color:#E8A838;font-weight:bold;margin-top:20pt;">Next: Scale to 16+ tissue types via STHELAR. Universal DAPI classifier.</p>
</body></html>`;
  fs.writeFileSync(path.join(SLIDES_DIR, 's26.html'), s26html);
  await html2pptx(path.join(SLIDES_DIR, 's26.html'), pptx);

  // Save
  await pptx.writeFile({ fileName: OUT });
  console.log(`Presentation saved to: ${OUT}`);
}

build().catch(e => { console.error(e); process.exit(1); });
