const sharp = require('sharp');
const React = require('react');
const ReactDOMServer = require('react-dom/server');
const { FaMicroscope, FaClock, FaMoneyBillWave, FaRobot, FaBullseye, FaTachometerAlt } = require('react-icons/fa');
const path = require('path');

const FIGS = path.join(__dirname, 'figures');

const SCALE = 3; // 3x resolution for high-DPI output

async function rasterizeIcon(IconComponent, color, size, filename) {
  const hiSize = size * SCALE;
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color: `#${color}`, size: String(hiSize) })
  );
  await sharp(Buffer.from(svg)).png().toFile(path.join(FIGS, filename));
}

async function createSvgPng(svgContent, filename, width, height) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width * SCALE}" height="${height * SCALE}" viewBox="0 0 ${width} ${height}">${svgContent}</svg>`;
  await sharp(Buffer.from(svg)).png().toFile(path.join(FIGS, filename));
}

async function main() {
  // Icons for comparison slide
  await rasterizeIcon(FaMicroscope, '00526B', 128, 'icon_microscope.png');
  await rasterizeIcon(FaClock, '00526B', 128, 'icon_clock.png');
  await rasterizeIcon(FaMoneyBillWave, '00526B', 128, 'icon_cost.png');
  await rasterizeIcon(FaRobot, '0089A7', 128, 'icon_robot.png');
  await rasterizeIcon(FaBullseye, '0089A7', 128, 'icon_target.png');
  await rasterizeIcon(FaTachometerAlt, '0089A7', 128, 'icon_speed.png');

  // Pipeline flow diagram
  await createSvgPng(`
    <defs>
      <marker id="arr" viewBox="0 0 10 6" refX="10" refY="3" markerWidth="8" markerHeight="6" orient="auto">
        <path d="M0,0 L10,3 L0,6 Z" fill="#00526B"/>
      </marker>
    </defs>
    <style>
      text { font-family: Arial, sans-serif; fill: white; font-weight: bold; }
      .label { fill: #003B4F; font-size: 11px; font-weight: normal; }
      .stage { fill: #003B4F; font-size: 12px; font-weight: bold; }
    </style>
    <!-- Input boxes -->
    <rect x="20" y="15" width="120" height="40" rx="6" fill="#00526B"/>
    <text x="80" y="40" text-anchor="middle" font-size="13">XENIUM</text>
    <rect x="160" y="15" width="120" height="40" rx="6" fill="#00526B"/>
    <text x="220" y="40" text-anchor="middle" font-size="13">MERSCOPE</text>
    <rect x="300" y="15" width="120" height="40" rx="6" fill="#00526B"/>
    <text x="360" y="40" text-anchor="middle" font-size="13">STHELAR</text>
    <!-- Arrows down -->
    <line x1="80" y1="55" x2="220" y2="85" stroke="#00526B" stroke-width="2" marker-end="url(#arr)"/>
    <line x1="220" y1="55" x2="220" y2="85" stroke="#00526B" stroke-width="2" marker-end="url(#arr)"/>
    <line x1="360" y1="55" x2="220" y2="85" stroke="#00526B" stroke-width="2" marker-end="url(#arr)"/>
    <!-- Annotate -->
    <rect x="110" y="90" width="220" height="50" rx="8" fill="#E8A838"/>
    <text x="220" y="112" text-anchor="middle" fill="#003B4F" font-size="14">ANNOTATE</text>
    <text x="220" y="130" text-anchor="middle" fill="#003B4F" font-size="10" font-weight="normal">11 Methods → Consensus</text>
    <!-- Arrow -->
    <line x1="220" y1="140" x2="220" y2="160" stroke="#00526B" stroke-width="2" marker-end="url(#arr)"/>
    <!-- Standardize -->
    <rect x="110" y="165" width="220" height="40" rx="8" fill="#F0C75E"/>
    <text x="220" y="190" text-anchor="middle" fill="#003B4F" font-size="13">STANDARDIZE (Cell Ontology)</text>
    <!-- Arrow -->
    <line x1="220" y1="205" x2="220" y2="225" stroke="#00526B" stroke-width="2" marker-end="url(#arr)"/>
    <!-- Filter -->
    <rect x="110" y="230" width="220" height="40" rx="8" fill="#7BC8A4"/>
    <text x="220" y="255" text-anchor="middle" fill="#003B4F" font-size="13">CONFIDENCE FILTERING</text>
    <!-- Arrow -->
    <line x1="220" y1="270" x2="220" y2="290" stroke="#00526B" stroke-width="2" marker-end="url(#arr)"/>
    <!-- Extract LMDB -->
    <rect x="110" y="295" width="220" height="40" rx="8" fill="#5DADE2"/>
    <text x="220" y="320" text-anchor="middle" fill="white" font-size="13">EXTRACT (LMDB Patches)</text>
    <!-- Arrow -->
    <line x1="220" y1="335" x2="220" y2="355" stroke="#00526B" stroke-width="2" marker-end="url(#arr)"/>
    <!-- Train -->
    <rect x="110" y="360" width="220" height="40" rx="8" fill="#A569BD"/>
    <text x="220" y="385" text-anchor="middle" fill="white" font-size="14">TRAIN (CNN / ViT)</text>
    <!-- Arrow -->
    <line x1="220" y1="400" x2="220" y2="420" stroke="#00526B" stroke-width="2" marker-end="url(#arr)"/>
    <!-- Output -->
    <rect x="130" y="425" width="180" height="40" rx="8" fill="#00526B"/>
    <text x="220" y="450" text-anchor="middle" fill="white" font-size="14">OUTPUT: MODEL</text>
    <!-- Side labels -->
    <text x="370" y="115" class="stage">Training Data</text>
    <text x="370" y="135" class="label">Generation</text>
    <text x="370" y="190" class="label">Label harmonization</text>
    <text x="370" y="255" class="label">Quality control</text>
    <text x="370" y="320" class="label">GPU-optimized format</text>
    <text x="370" y="385" class="stage">Model Training</text>
    <text x="370" y="450" class="stage">Deployment</text>
  `, 'pipeline_flow.png', 500, 480);

  // Confidence tiers — bar widths proportional to agreement %
  // Full bar = 420px = 100%. Tier2 80% = 336px. Tier3 50% = 210px.
  await createSvgPng(`
    <style>text { font-family: Arial, sans-serif; }</style>
    <!-- Tier 1: 100% -->
    <text x="20" y="30" font-size="16" font-weight="bold" fill="#003B4F">Tier 1: High Confidence</text>
    <rect x="20" y="40" width="420" height="35" rx="6" fill="#2ECC40"/>
    <text x="30" y="63" font-size="13" fill="white" font-weight="bold">100% Algorithm Agreement</text>
    <!-- Tier 2: >80% -->
    <text x="20" y="110" font-size="16" font-weight="bold" fill="#003B4F">Tier 2: Medium Confidence</text>
    <rect x="20" y="120" width="420" height="35" rx="6" fill="#FFF3CD"/>
    <rect x="20" y="120" width="336" height="35" rx="6" fill="#FFD700"/>
    <text x="30" y="143" font-size="13" fill="#003B4F" font-weight="bold">&gt;80% Agreement</text>
    <!-- Tier 3: >50% -->
    <text x="20" y="190" font-size="16" font-weight="bold" fill="#003B4F">Tier 3: Low Confidence</text>
    <rect x="20" y="200" width="420" height="35" rx="6" fill="#FFE0CC"/>
    <rect x="20" y="200" width="210" height="35" rx="6" fill="#FF851B"/>
    <text x="30" y="223" font-size="13" fill="white" font-weight="bold">&gt;50% Agreement</text>
    <!-- Discarded -->
    <text x="140" y="275" font-size="14" fill="#CC0000" font-weight="bold">Discarded (&lt;50% Agreement)</text>
  `, 'confidence_tiers.png', 460, 300);

  // Hierarchy tree — TME-focused with CL nomenclature
  // 5 broad, ~13 medium, ~15 fine. Wide canvas for spacing.
  await createSvgPng(`
    <style>
      text{font-family:Arial,sans-serif}
      .c{stroke:#aaa;stroke-width:1;fill:none}
      .lv{font-weight:bold}
    </style>
    <!-- ══ BROAD (y=20, h=30) ══ -->
    <text x="8" y="40" font-size="13" class="lv" fill="#CC0000">Broad</text>
    <rect x="72" y="18" width="88" height="30" rx="6" fill="#5B9F3F"/>
    <text x="116" y="38" text-anchor="middle" font-size="11" fill="white" font-weight="bold">Epithelial</text>
    <rect x="188" y="18" width="200" height="30" rx="6" fill="#00526B"/>
    <text x="288" y="38" text-anchor="middle" font-size="11" fill="white" font-weight="bold">Immune</text>
    <rect x="416" y="18" width="80" height="30" rx="6" fill="#7D6B57"/>
    <text x="456" y="38" text-anchor="middle" font-size="11" fill="white" font-weight="bold">Stromal</text>
    <rect x="524" y="18" width="95" height="30" rx="6" fill="#2E86C1"/>
    <text x="571" y="38" text-anchor="middle" font-size="11" fill="white" font-weight="bold">Endothelial</text>
    <rect x="647" y="18" width="90" height="30" rx="6" fill="#922B21"/>
    <text x="692" y="38" text-anchor="middle" font-size="11" fill="white" font-weight="bold">Neoplastic</text>
    <!-- ══ BROAD→MEDIUM connectors ══ -->
    <path d="M100,48 L100,65 L80,75" class="c"/>
    <path d="M132,48 L132,65 L155,75" class="c"/>
    <path d="M230,48 L230,65 L225,75" class="c"/>
    <path d="M265,48 L265,65 L290,75" class="c"/>
    <path d="M300,48 L300,65 L353,75" class="c"/>
    <path d="M320,48 L320,65 L406,75" class="c"/>
    <path d="M345,48 L345,65 L462,75" class="c"/>
    <path d="M365,48 L365,65 L510,75" class="c"/>
    <path d="M440,48 L440,65 L560,75" class="c"/>
    <path d="M472,48 L472,65 L622,75" class="c"/>
    <path d="M550,48 L550,65 L670,75" class="c"/>
    <path d="M592,48 L592,65 L735,75" class="c"/>
    <path d="M692,48 L692,65 L792,75" class="c"/>
    <!-- ══ MEDIUM (y=77, h=26) ══ -->
    <text x="8" y="96" font-size="13" class="lv" fill="#E8A838">Medium</text>
    <rect x="58" y="77" width="55" height="26" rx="5" fill="#FFF3CD" stroke="#E8A838" stroke-width="1"/>
    <text x="85" y="94" text-anchor="middle" font-size="9" fill="#003B4F">Luminal</text>
    <rect x="120" y="77" width="78" height="26" rx="5" fill="#FFF3CD" stroke="#E8A838" stroke-width="1"/>
    <text x="159" y="94" text-anchor="middle" font-size="9" fill="#003B4F">Myoepithelial</text>
    <rect x="206" y="77" width="48" height="26" rx="5" fill="#D6EAF8" stroke="#2E86C1" stroke-width="1"/>
    <text x="230" y="94" text-anchor="middle" font-size="9" fill="#003B4F">T cell</text>
    <rect x="262" y="77" width="48" height="26" rx="5" fill="#D6EAF8" stroke="#2E86C1" stroke-width="1"/>
    <text x="286" y="94" text-anchor="middle" font-size="9" fill="#003B4F">B cell</text>
    <rect x="318" y="77" width="60" height="26" rx="5" fill="#D6EAF8" stroke="#2E86C1" stroke-width="1"/>
    <text x="348" y="94" text-anchor="middle" font-size="9" fill="#003B4F">NK cell</text>
    <rect x="386" y="77" width="50" height="26" rx="5" fill="#D6EAF8" stroke="#2E86C1" stroke-width="1"/>
    <text x="411" y="94" text-anchor="middle" font-size="8" fill="#003B4F">Macrophage</text>
    <rect x="444" y="77" width="42" height="26" rx="5" fill="#D6EAF8" stroke="#2E86C1" stroke-width="1"/>
    <text x="465" y="94" text-anchor="middle" font-size="9" fill="#003B4F">DC</text>
    <rect x="494" y="77" width="42" height="26" rx="5" fill="#D6EAF8" stroke="#2E86C1" stroke-width="1"/>
    <text x="515" y="94" text-anchor="middle" font-size="8" fill="#003B4F">Monocyte</text>
    <rect x="544" y="77" width="42" height="26" rx="5" fill="#F5E6D3" stroke="#7D6B57" stroke-width="1"/>
    <text x="565" y="94" text-anchor="middle" font-size="8" fill="#003B4F">Fibroblast</text>
    <rect x="594" y="77" width="50" height="26" rx="5" fill="#F5E6D3" stroke="#7D6B57" stroke-width="1"/>
    <text x="619" y="94" text-anchor="middle" font-size="9" fill="#003B4F">Pericyte</text>
    <rect x="652" y="77" width="48" height="26" rx="5" fill="#D6EAF8" stroke="#2E86C1" stroke-width="1"/>
    <text x="676" y="94" text-anchor="middle" font-size="8" fill="#003B4F">Vascular</text>
    <rect x="708" y="77" width="56" height="26" rx="5" fill="#D6EAF8" stroke="#2E86C1" stroke-width="1"/>
    <text x="736" y="94" text-anchor="middle" font-size="8" fill="#003B4F">Lymphatic</text>
    <rect x="772" y="77" width="58" height="26" rx="5" fill="#F5B7B1" stroke="#922B21" stroke-width="1"/>
    <text x="801" y="94" text-anchor="middle" font-size="8" fill="#003B4F">Malignant</text>
    <!-- ══ MEDIUM→FINE connectors (only for T, B, Macro, DC, Fibro) ══ -->
    <!-- T cell fan -->
    <path d="M222,103 L222,118 L112,130" class="c"/>
    <path d="M230,103 L230,118 L172,130" class="c"/>
    <path d="M237,103 L237,118 L232,130" class="c"/>
    <path d="M237,103 L237,118 L292,130" class="c"/>
    <path d="M237,103 L237,118 L352,130" class="c"/>
    <!-- B cell fan -->
    <path d="M280,103 L280,118 L402,130" class="c"/>
    <path d="M286,103 L286,118 L452,130" class="c"/>
    <path d="M292,103 L292,118 L502,130" class="c"/>
    <!-- Macrophage fan -->
    <path d="M405,103 L405,118 L548,130" class="c"/>
    <path d="M417,103 L417,118 L580,130" class="c"/>
    <!-- DC fan -->
    <path d="M458,103 L458,118 L618,130" class="c"/>
    <path d="M472,103 L472,118 L658,130" class="c"/>
    <path d="M472,103 L472,118 L698,130" class="c"/>
    <!-- Fibroblast -->
    <path d="M565,103 L565,118 L748,130" class="c"/>
    <path d="M565,103 L565,118 L808,130" class="c"/>
    <!-- ══ FINE (y=132, h=22) ══ -->
    <text x="8" y="148" font-size="13" class="lv" fill="#2ECC40">Fine</text>
    <rect x="82" y="132" width="60" height="22" rx="4" fill="#D5F5E3" stroke="#82B366" stroke-width="1"/>
    <text x="112" y="147" text-anchor="middle" font-size="8" fill="#003B4F">CD4+ helper</text>
    <rect x="148" y="132" width="48" height="22" rx="4" fill="#D5F5E3" stroke="#82B366" stroke-width="1"/>
    <text x="172" y="147" text-anchor="middle" font-size="8" fill="#003B4F">Treg</text>
    <rect x="202" y="132" width="60" height="22" rx="4" fill="#D5F5E3" stroke="#82B366" stroke-width="1"/>
    <text x="232" y="147" text-anchor="middle" font-size="8" fill="#003B4F">CD8+ cyto</text>
    <rect x="268" y="132" width="48" height="22" rx="4" fill="#D5F5E3" stroke="#82B366" stroke-width="1"/>
    <text x="292" y="147" text-anchor="middle" font-size="7" fill="#003B4F">CD8+ exh</text>
    <rect x="322" y="132" width="60" height="22" rx="4" fill="#D5F5E3" stroke="#82B366" stroke-width="1"/>
    <text x="352" y="147" text-anchor="middle" font-size="8" fill="#003B4F">γδ T cell</text>
    <rect x="388" y="132" width="38" height="22" rx="4" fill="#D5F5E3" stroke="#82B366" stroke-width="1"/>
    <text x="407" y="147" text-anchor="middle" font-size="7" fill="#003B4F">Naive B</text>
    <rect x="432" y="132" width="46" height="22" rx="4" fill="#D5F5E3" stroke="#82B366" stroke-width="1"/>
    <text x="455" y="147" text-anchor="middle" font-size="7" fill="#003B4F">Memory B</text>
    <rect x="484" y="132" width="42" height="22" rx="4" fill="#D5F5E3" stroke="#82B366" stroke-width="1"/>
    <text x="505" y="147" text-anchor="middle" font-size="8" fill="#003B4F">Plasma</text>
    <rect x="532" y="132" width="28" height="22" rx="4" fill="#D5F5E3" stroke="#82B366" stroke-width="1"/>
    <text x="546" y="147" text-anchor="middle" font-size="8" fill="#003B4F">M1</text>
    <rect x="566" y="132" width="28" height="22" rx="4" fill="#D5F5E3" stroke="#82B366" stroke-width="1"/>
    <text x="580" y="147" text-anchor="middle" font-size="8" fill="#003B4F">M2</text>
    <rect x="600" y="132" width="38" height="22" rx="4" fill="#D5F5E3" stroke="#82B366" stroke-width="1"/>
    <text x="619" y="147" text-anchor="middle" font-size="8" fill="#003B4F">cDC1</text>
    <rect x="644" y="132" width="38" height="22" rx="4" fill="#D5F5E3" stroke="#82B366" stroke-width="1"/>
    <text x="663" y="147" text-anchor="middle" font-size="8" fill="#003B4F">cDC2</text>
    <rect x="688" y="132" width="28" height="22" rx="4" fill="#D5F5E3" stroke="#82B366" stroke-width="1"/>
    <text x="702" y="147" text-anchor="middle" font-size="8" fill="#003B4F">pDC</text>
    <rect x="722" y="132" width="52" height="22" rx="4" fill="#D5F5E3" stroke="#82B366" stroke-width="1"/>
    <text x="748" y="147" text-anchor="middle" font-size="7" fill="#003B4F">Myofibro</text>
    <rect x="780" y="132" width="52" height="22" rx="4" fill="#D5F5E3" stroke="#82B366" stroke-width="1"/>
    <text x="806" y="147" text-anchor="middle" font-size="8" fill="#003B4F">CAF</text>
    <!-- ══ Class counts ══ -->
    <text x="110" y="173" font-size="9" fill="#CC0000" font-weight="bold">5 classes</text>
    <text x="350" y="173" font-size="9" fill="#E8A838" font-weight="bold">~15 classes</text>
    <text x="580" y="173" font-size="9" fill="#2ECC40" font-weight="bold">~50 classes (Cell Ontology)</text>
  `, 'hierarchy_tree.png', 850, 185);

  // Ensemble funnel
  await createSvgPng(`
    <style>text { font-family: Arial, sans-serif; }</style>
    <defs>
      <linearGradient id="funnel" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" style="stop-color:#E8F4FD"/>
        <stop offset="100%" style="stop-color:#B3D9F2"/>
      </linearGradient>
    </defs>
    <!-- Funnel shape -->
    <path d="M60,10 L440,10 L440,50 Q440,60 430,65 L320,130 Q310,135 310,145 L310,190 L190,190 L190,145 Q190,135 180,130 L70,65 Q60,60 60,50 Z" fill="url(#funnel)" stroke="#0089A7" stroke-width="2"/>
    <!-- Method names in funnel -->
    <text x="120" y="35" font-size="11" fill="#0066CC" font-weight="bold">CellTypist</text>
    <text x="230" y="30" font-size="11" fill="#003B4F" font-weight="bold">SingleR</text>
    <text x="320" y="35" font-size="11" fill="#0066CC" font-weight="bold">ScType</text>
    <text x="140" y="55" font-size="10" fill="#006633">SCINA</text>
    <text x="220" y="55" font-size="10" fill="#006633">PopV</text>
    <text x="290" y="55" font-size="10" fill="#006633">Azimuth</text>
    <text x="370" y="42" font-size="10" fill="#006633">ScANVI</text>
    <text x="160" y="75" font-size="10" fill="#999">ScArches</text>
    <text x="250" y="80" font-size="10" fill="#999">Clustifyr</text>
    <text x="320" y="70" font-size="10" fill="#999">Garnett</text>
    <text x="200" y="100" font-size="10" fill="#999">ML</text>
    <!-- Output box -->
    <rect x="175" y="195" width="150" height="35" rx="6" fill="#00526B"/>
    <text x="250" y="217" text-anchor="middle" font-size="13" fill="white" font-weight="bold">Consensus Label</text>
  `, 'ensemble_funnel.png', 500, 245);

  // Training curve — realistic curriculum learning with phase transition dips
  // When switching from broad→medium, new head unfreezes → temporary accuracy drop
  // and loss spike as model adapts to finer categories. Same for medium→fine.
  await createSvgPng(`
    <style>text{font-family:Arial,sans-serif}</style>
    <!-- Axes -->
    <line x1="60" y1="25" x2="60" y2="280" stroke="#003B4F" stroke-width="2"/>
    <line x1="60" y1="280" x2="530" y2="280" stroke="#003B4F" stroke-width="2"/>
    <text x="30" y="155" font-size="12" fill="#003B4F" transform="rotate(-90,30,155)">Performance</text>
    <text x="295" y="310" text-anchor="middle" font-size="12" fill="#003B4F">Epochs</text>
    <!-- Y-axis labels -->
    <text x="55" y="280" text-anchor="end" font-size="9" fill="#999">0</text>
    <text x="55" y="218" text-anchor="end" font-size="9" fill="#999">0.2</text>
    <text x="55" y="155" text-anchor="end" font-size="9" fill="#999">0.5</text>
    <text x="55" y="93" text-anchor="end" font-size="9" fill="#999">0.8</text>
    <text x="55" y="35" text-anchor="end" font-size="9" fill="#999">1.0</text>
    <!-- Gridlines -->
    <line x1="60" y1="218" x2="520" y2="218" stroke="#eee" stroke-width="0.5"/>
    <line x1="60" y1="155" x2="520" y2="155" stroke="#eee" stroke-width="0.5"/>
    <line x1="60" y1="93" x2="520" y2="93" stroke="#eee" stroke-width="0.5"/>
    <!-- Phase bars at top -->
    <rect x="60" y="8" width="150" height="14" fill="#CC0000" rx="3"/>
    <text x="135" y="19" text-anchor="middle" font-size="10" fill="white" font-weight="bold">Broad (3 classes)</text>
    <rect x="210" y="8" width="140" height="14" fill="#E8A838" rx="3"/>
    <text x="280" y="19" text-anchor="middle" font-size="10" fill="white" font-weight="bold">Medium (~15)</text>
    <rect x="350" y="8" width="140" height="14" fill="#2ECC40" rx="3"/>
    <text x="420" y="19" text-anchor="middle" font-size="10" fill="white" font-weight="bold">Fine (~50)</text>
    <!-- Phase transition lines -->
    <line x1="210" y1="25" x2="210" y2="280" stroke="#CC0000" stroke-width="1.5" stroke-dasharray="6,3" opacity="0.4"/>
    <line x1="350" y1="25" x2="350" y2="280" stroke="#E8A838" stroke-width="1.5" stroke-dasharray="6,3" opacity="0.4"/>
    <!-- ═══ VALIDATION ACCURACY (green) ═══ -->
    <!-- Broad phase: rapid rise to ~0.88 -->
    <path d="M65,260 Q90,230 120,130 Q150,85 180,72 Q195,68 208,65" fill="none" stroke="#2ECC40" stroke-width="3"/>
    <!-- Transition dip: drops when medium head activates -->
    <path d="M208,65 Q215,110 220,120" fill="none" stroke="#2ECC40" stroke-width="3"/>
    <!-- Medium phase: recovers and climbs -->
    <path d="M220,120 Q240,100 270,82 Q300,68 330,60 Q340,58 348,56" fill="none" stroke="#2ECC40" stroke-width="3"/>
    <!-- Transition dip: drops when fine head activates -->
    <path d="M348,56 Q355,95 360,105" fill="none" stroke="#2ECC40" stroke-width="3"/>
    <!-- Fine phase: slower recovery, plateaus lower -->
    <path d="M360,105 Q380,85 410,72 Q440,62 470,56 Q490,53 510,52" fill="none" stroke="#2ECC40" stroke-width="3"/>
    <!-- ═══ TRAINING LOSS (blue) ═══ -->
    <!-- Broad phase: drops fast -->
    <path d="M65,40 Q90,60 120,140 Q150,185 180,210 Q195,218 208,220" fill="none" stroke="#0066CC" stroke-width="3"/>
    <!-- Transition spike: loss jumps when new head unfreezes -->
    <path d="M208,220 Q215,170 220,160" fill="none" stroke="#0066CC" stroke-width="3"/>
    <!-- Medium phase: drops again -->
    <path d="M220,160 Q240,190 270,215 Q300,235 330,245 Q340,248 348,250" fill="none" stroke="#0066CC" stroke-width="3"/>
    <!-- Transition spike -->
    <path d="M348,250 Q355,200 360,190" fill="none" stroke="#0066CC" stroke-width="3"/>
    <!-- Fine phase: settles higher than before -->
    <path d="M360,190 Q380,215 410,235 Q440,248 470,255 Q490,258 510,260" fill="none" stroke="#0066CC" stroke-width="3"/>
    <!-- Curve labels -->
    <text x="515" y="50" font-size="10" fill="#2ECC40" font-weight="bold">Val F1</text>
    <text x="515" y="264" font-size="10" fill="#0066CC" font-weight="bold">Train Loss</text>
    <!-- Transition annotations -->
    <text x="212" y="145" font-size="8" fill="#CC0000" font-weight="bold">dip</text>
    <path d="M215,135 L215,124" stroke="#CC0000" stroke-width="1" marker-end="url(#arr)"/>
    <text x="352" y="130" font-size="8" fill="#E8A838" font-weight="bold">dip</text>
    <path d="M355,120 L355,109" stroke="#E8A838" stroke-width="1" marker-end="url(#arr)"/>
    <!-- Early stopping marker -->
    <rect x="490" y="25" width="30" height="255" fill="rgba(204,0,0,0.06)"/>
    <text x="505" y="155" font-size="8" fill="#CC0000" transform="rotate(-90,505,155)" font-weight="bold">Early Stop</text>
  `, 'training_curve.png', 570, 320);

  console.log('All assets generated!');
}

main().catch(console.error);
