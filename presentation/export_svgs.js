const fs = require('fs');
const path = require('path');
const React = require('react');
const ReactDOMServer = require('react-dom/server');
const { FaMicroscope, FaClock, FaMoneyBillWave, FaRobot, FaBullseye, FaTachometerAlt } = require('react-icons/fa');

const OUT = path.join(__dirname, 'figures-svg');

function saveSvg(content, filename, width, height) {
  const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
${content}
</svg>`;
  fs.writeFileSync(path.join(OUT, filename), svg);
}

function saveIcon(IconComponent, color, size, filename) {
  const markup = ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color: `#${color}`, size: String(size) })
  );
  // Wrap in proper SVG with xml declaration
  const svg = `<?xml version="1.0" encoding="UTF-8"?>\n${markup}`;
  fs.writeFileSync(path.join(OUT, filename), svg);
}

// ── Icons ──
saveIcon(FaMicroscope, '00526B', 128, 'icon_microscope.svg');
saveIcon(FaClock, '00526B', 128, 'icon_clock.svg');
saveIcon(FaMoneyBillWave, '00526B', 128, 'icon_cost.svg');
saveIcon(FaRobot, '0089A7', 128, 'icon_robot.svg');
saveIcon(FaBullseye, '0089A7', 128, 'icon_target.svg');
saveIcon(FaTachometerAlt, '0089A7', 128, 'icon_speed.svg');

// ── Pipeline Flow ──
saveSvg(`
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
  <rect x="20" y="15" width="120" height="40" rx="6" fill="#00526B"/>
  <text x="80" y="40" text-anchor="middle" font-size="13">XENIUM</text>
  <rect x="160" y="15" width="120" height="40" rx="6" fill="#00526B"/>
  <text x="220" y="40" text-anchor="middle" font-size="13">MERSCOPE</text>
  <rect x="300" y="15" width="120" height="40" rx="6" fill="#00526B"/>
  <text x="360" y="40" text-anchor="middle" font-size="13">STHELAR</text>
  <line x1="80" y1="55" x2="220" y2="85" stroke="#00526B" stroke-width="2" marker-end="url(#arr)"/>
  <line x1="220" y1="55" x2="220" y2="85" stroke="#00526B" stroke-width="2" marker-end="url(#arr)"/>
  <line x1="360" y1="55" x2="220" y2="85" stroke="#00526B" stroke-width="2" marker-end="url(#arr)"/>
  <rect x="110" y="90" width="220" height="50" rx="8" fill="#E8A838"/>
  <text x="220" y="112" text-anchor="middle" fill="#003B4F" font-size="14">ANNOTATE</text>
  <text x="220" y="130" text-anchor="middle" fill="#003B4F" font-size="10" font-weight="normal">11 Methods → Consensus</text>
  <line x1="220" y1="140" x2="220" y2="160" stroke="#00526B" stroke-width="2" marker-end="url(#arr)"/>
  <rect x="110" y="165" width="220" height="40" rx="8" fill="#F0C75E"/>
  <text x="220" y="190" text-anchor="middle" fill="#003B4F" font-size="13">STANDARDIZE (Cell Ontology)</text>
  <line x1="220" y1="205" x2="220" y2="225" stroke="#00526B" stroke-width="2" marker-end="url(#arr)"/>
  <rect x="110" y="230" width="220" height="40" rx="8" fill="#7BC8A4"/>
  <text x="220" y="255" text-anchor="middle" fill="#003B4F" font-size="13">CONFIDENCE FILTERING</text>
  <line x1="220" y1="270" x2="220" y2="290" stroke="#00526B" stroke-width="2" marker-end="url(#arr)"/>
  <rect x="110" y="295" width="220" height="40" rx="8" fill="#5DADE2"/>
  <text x="220" y="320" text-anchor="middle" fill="white" font-size="13">EXTRACT (LMDB Patches)</text>
  <line x1="220" y1="335" x2="220" y2="355" stroke="#00526B" stroke-width="2" marker-end="url(#arr)"/>
  <rect x="110" y="360" width="220" height="40" rx="8" fill="#A569BD"/>
  <text x="220" y="385" text-anchor="middle" fill="white" font-size="14">TRAIN (CNN / ViT)</text>
  <line x1="220" y1="400" x2="220" y2="420" stroke="#00526B" stroke-width="2" marker-end="url(#arr)"/>
  <rect x="130" y="425" width="180" height="40" rx="8" fill="#00526B"/>
  <text x="220" y="450" text-anchor="middle" fill="white" font-size="14">OUTPUT: MODEL</text>
  <text x="370" y="115" class="stage">Training Data</text>
  <text x="370" y="135" class="label">Generation</text>
  <text x="370" y="190" class="label">Label harmonization</text>
  <text x="370" y="255" class="label">Quality control</text>
  <text x="370" y="320" class="label">GPU-optimized format</text>
  <text x="370" y="385" class="stage">Model Training</text>
  <text x="370" y="450" class="stage">Deployment</text>
`, 'pipeline_flow.svg', 500, 480);

// ── Confidence Tiers ──
saveSvg(`
  <style>text { font-family: Arial, sans-serif; }</style>
  <text x="20" y="30" font-size="16" font-weight="bold" fill="#003B4F">Tier 1: High Confidence</text>
  <rect x="20" y="40" width="420" height="35" rx="6" fill="#2ECC40"/>
  <text x="30" y="63" font-size="13" fill="white" font-weight="bold">100% Algorithm Agreement</text>
  <text x="20" y="110" font-size="16" font-weight="bold" fill="#003B4F">Tier 2: Medium Confidence</text>
  <rect x="20" y="120" width="420" height="35" rx="6" fill="#FFF3CD"/>
  <rect x="20" y="120" width="336" height="35" rx="6" fill="#FFD700"/>
  <text x="30" y="143" font-size="13" fill="#003B4F" font-weight="bold">&gt;80% Agreement</text>
  <text x="20" y="190" font-size="16" font-weight="bold" fill="#003B4F">Tier 3: Low Confidence</text>
  <rect x="20" y="200" width="420" height="35" rx="6" fill="#FFE0CC"/>
  <rect x="20" y="200" width="210" height="35" rx="6" fill="#FF851B"/>
  <text x="30" y="223" font-size="13" fill="white" font-weight="bold">&gt;50% Agreement</text>
  <text x="140" y="275" font-size="14" fill="#CC0000" font-weight="bold">Discarded (&lt;50% Agreement)</text>
`, 'confidence_tiers.svg', 460, 300);

// ── Hierarchy Tree ──
saveSvg(`
  <style>
    text { font-family: Arial, sans-serif; }
    .box { rx: 8; ry: 8; }
    .conn { stroke: #888; stroke-width: 1.5; fill: none; }
    .lvl { font-size: 15px; font-weight: bold; }
  </style>
  <text x="10" y="42" class="lvl" fill="#CC0000">Broad</text>
  <rect x="100" y="18" width="110" height="38" class="box" fill="#D5E8D4" stroke="#82B366" stroke-width="1.5"/>
  <text x="155" y="43" text-anchor="middle" font-size="13" fill="#003B4F" font-weight="bold">Epithelial</text>
  <rect x="280" y="18" width="110" height="38" class="box" fill="#00526B"/>
  <text x="335" y="43" text-anchor="middle" font-size="13" fill="white" font-weight="bold">Immune</text>
  <rect x="460" y="18" width="110" height="38" class="box" fill="#D5E8D4" stroke="#82B366" stroke-width="1.5"/>
  <text x="515" y="43" text-anchor="middle" font-size="13" fill="#003B4F" font-weight="bold">Stromal</text>
  <path d="M155,56 L155,78 L110,95" class="conn"/>
  <path d="M155,56 L155,78 L205,95" class="conn"/>
  <path d="M335,56 L335,78 L305,95" class="conn"/>
  <path d="M335,56 L335,78 L390,95" class="conn"/>
  <path d="M515,56 L515,78 L485,95" class="conn"/>
  <path d="M515,56 L515,78 L585,95" class="conn"/>
  <text x="10" y="120" class="lvl" fill="#E8A838">Medium</text>
  <rect x="65" y="95" width="90" height="32" class="box" fill="#FFF3CD" stroke="#E8A838" stroke-width="1.5"/>
  <text x="110" y="116" text-anchor="middle" font-size="12" fill="#003B4F">Luminal</text>
  <rect x="165" y="95" width="90" height="32" class="box" fill="#FFF3CD" stroke="#E8A838" stroke-width="1.5"/>
  <text x="210" y="116" text-anchor="middle" font-size="12" fill="#003B4F">Basal</text>
  <rect x="265" y="95" width="90" height="32" class="box" fill="#FFF3CD" stroke="#E8A838" stroke-width="1.5"/>
  <text x="310" y="116" text-anchor="middle" font-size="12" fill="#003B4F">T-Cell</text>
  <rect x="340" y="95" width="90" height="32" class="box" fill="#FFF3CD" stroke="#E8A838" stroke-width="1.5"/>
  <text x="385" y="116" text-anchor="middle" font-size="12" fill="#003B4F">B-Cell</text>
  <rect x="435" y="95" width="90" height="32" class="box" fill="#FFF3CD" stroke="#E8A838" stroke-width="1.5"/>
  <text x="480" y="116" text-anchor="middle" font-size="12" fill="#003B4F">Fibroblast</text>
  <rect x="535" y="95" width="90" height="32" class="box" fill="#FFF3CD" stroke="#E8A838" stroke-width="1.5"/>
  <text x="580" y="116" text-anchor="middle" font-size="12" fill="#003B4F">Pericyte</text>
  <path d="M310,127 L310,150 L275,168" class="conn"/>
  <path d="M310,127 L310,150 L365,168" class="conn"/>
  <text x="10" y="195" class="lvl" fill="#2ECC40">Fine</text>
  <rect x="230" y="168" width="90" height="32" class="box" fill="#D5E8D4" stroke="#82B366" stroke-width="1.5"/>
  <text x="275" y="189" text-anchor="middle" font-size="12" fill="#003B4F">CD4+</text>
  <rect x="320" y="168" width="90" height="32" class="box" fill="#D5E8D4" stroke="#82B366" stroke-width="1.5"/>
  <text x="365" y="189" text-anchor="middle" font-size="12" fill="#003B4F">CD8+</text>
`, 'hierarchy_tree.svg', 670, 220);

// ── Ensemble Funnel ──
saveSvg(`
  <style>text { font-family: Arial, sans-serif; }</style>
  <defs>
    <linearGradient id="funnel" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#E8F4FD"/>
      <stop offset="100%" style="stop-color:#B3D9F2"/>
    </linearGradient>
  </defs>
  <path d="M60,10 L440,10 L440,50 Q440,60 430,65 L320,130 Q310,135 310,145 L310,190 L190,190 L190,145 Q190,135 180,130 L70,65 Q60,60 60,50 Z" fill="url(#funnel)" stroke="#0089A7" stroke-width="2"/>
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
  <rect x="175" y="195" width="150" height="35" rx="6" fill="#00526B"/>
  <text x="250" y="217" text-anchor="middle" font-size="13" fill="white" font-weight="bold">Consensus Label</text>
`, 'ensemble_funnel.svg', 500, 245);

// ── Training Curve ──
saveSvg(`
  <style>text { font-family: Arial, sans-serif; }</style>
  <line x1="50" y1="10" x2="50" y2="230" stroke="#003B4F" stroke-width="2"/>
  <line x1="50" y1="230" x2="430" y2="230" stroke="#003B4F" stroke-width="2"/>
  <text x="25" y="125" font-size="12" fill="#003B4F" transform="rotate(-90,25,125)">Performance</text>
  <text x="240" y="255" text-anchor="middle" font-size="12" fill="#003B4F">Epochs</text>
  <rect x="50" y="5" width="120" height="12" fill="#CC0000" rx="3"/>
  <text x="110" y="14" text-anchor="middle" font-size="9" fill="white" font-weight="bold">Broad</text>
  <rect x="170" y="5" width="110" height="12" fill="#E8A838" rx="3"/>
  <text x="225" y="14" text-anchor="middle" font-size="9" fill="white" font-weight="bold">Medium</text>
  <rect x="280" y="5" width="110" height="12" fill="#2ECC40" rx="3"/>
  <text x="335" y="14" text-anchor="middle" font-size="9" fill="white" font-weight="bold">Fine</text>
  <path d="M55,210 Q100,200 140,170 Q185,140 230,100 Q280,70 330,55 Q370,47 400,43" fill="none" stroke="#2ECC40" stroke-width="3"/>
  <path d="M55,30 Q100,50 140,100 Q185,140 230,170 Q280,190 330,200 Q370,208 400,212" fill="none" stroke="#0066CC" stroke-width="3"/>
  <text x="410" y="40" font-size="10" fill="#2ECC40" font-weight="bold">Val Accuracy</text>
  <text x="410" y="218" font-size="10" fill="#0066CC" font-weight="bold">Train Error</text>
  <line x1="140" y1="20" x2="140" y2="230" stroke="#CCC" stroke-width="1" stroke-dasharray="4,4"/>
  <line x1="230" y1="20" x2="230" y2="230" stroke="#CCC" stroke-width="1" stroke-dasharray="4,4"/>
  <line x1="330" y1="20" x2="330" y2="230" stroke="#CCC" stroke-width="1" stroke-dasharray="4,4"/>
  <rect x="390" y="20" width="40" height="210" fill="rgba(204,0,0,0.08)"/>
  <text x="410" y="130" font-size="9" fill="#CC0000" transform="rotate(-90,410,130)" font-weight="bold">Overtraining</text>
`, 'training_curve.svg', 500, 270);

// ── CNN Architecture ──
saveSvg(`
  <defs>
    <marker id="arr" viewBox="0 0 10 6" refX="10" refY="3" markerWidth="8" markerHeight="6" orient="auto">
      <path d="M0,0 L10,3 L0,6 Z" fill="#00526B"/>
    </marker>
    <linearGradient id="convGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#5DADE2"/>
      <stop offset="100%" style="stop-color:#2E86C1"/>
    </linearGradient>
    <linearGradient id="poolGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#58D68D"/>
      <stop offset="100%" style="stop-color:#28B463"/>
    </linearGradient>
    <linearGradient id="fcGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#AF7AC5"/>
      <stop offset="100%" style="stop-color:#7D3C98"/>
    </linearGradient>
  </defs>
  <style>
    text { font-family: Arial, sans-serif; }
    .title { font-size: 18px; font-weight: bold; fill: #00526B; }
    .label { font-size: 11px; fill: #333; }
    .sublabel { font-size: 9px; fill: #666; }
    .dim { font-size: 9px; fill: #999; font-style: italic; }
    .arrow { stroke: #00526B; stroke-width: 2; fill: none; }
  </style>
  <text x="400" y="24" text-anchor="middle" class="title">Convolutional Neural Network (CNN) Architecture</text>
  <rect x="15" y="70" width="55" height="55" rx="4" fill="#E8F4FD" stroke="#5DADE2" stroke-width="1.5"/>
  <rect x="20" y="75" width="55" height="55" rx="4" fill="#D4EFFC" stroke="#5DADE2" stroke-width="1.5"/>
  <rect x="25" y="80" width="55" height="55" rx="4" fill="#BEE6F9" stroke="#5DADE2" stroke-width="1.5"/>
  <text x="52" y="115" text-anchor="middle" class="label" font-weight="bold">Input</text>
  <text x="52" y="155" text-anchor="middle" class="dim">128x128x3</text>
  <line x1="85" y1="107" x2="110" y2="107" class="arrow" marker-end="url(#arr)"/>
  <rect x="115" y="55" width="50" height="100" rx="5" fill="url(#convGrad)"/>
  <text x="140" y="100" text-anchor="middle" font-size="10" fill="white" font-weight="bold">Conv</text>
  <text x="140" y="113" text-anchor="middle" font-size="8" fill="#D6EAF8">3x3</text>
  <text x="140" y="170" text-anchor="middle" class="dim">64 filters</text>
  <rect x="170" y="75" width="25" height="60" rx="3" fill="url(#poolGrad)"/>
  <text x="182" y="110" text-anchor="middle" font-size="7" fill="white" font-weight="bold">Pool</text>
  <line x1="200" y1="107" x2="215" y2="107" class="arrow" marker-end="url(#arr)"/>
  <rect x="220" y="60" width="50" height="90" rx="5" fill="url(#convGrad)"/>
  <text x="245" y="100" text-anchor="middle" font-size="10" fill="white" font-weight="bold">Conv</text>
  <text x="245" y="113" text-anchor="middle" font-size="8" fill="#D6EAF8">3x3</text>
  <text x="245" y="170" text-anchor="middle" class="dim">128 filters</text>
  <rect x="275" y="75" width="25" height="60" rx="3" fill="url(#poolGrad)"/>
  <text x="287" y="110" text-anchor="middle" font-size="7" fill="white" font-weight="bold">Pool</text>
  <line x1="305" y1="107" x2="320" y2="107" class="arrow" marker-end="url(#arr)"/>
  <rect x="325" y="65" width="50" height="80" rx="5" fill="url(#convGrad)"/>
  <text x="350" y="100" text-anchor="middle" font-size="10" fill="white" font-weight="bold">Conv</text>
  <text x="350" y="113" text-anchor="middle" font-size="8" fill="#D6EAF8">3x3</text>
  <text x="350" y="170" text-anchor="middle" class="dim">256 filters</text>
  <rect x="380" y="70" width="50" height="70" rx="5" fill="url(#convGrad)" opacity="0.85"/>
  <text x="405" y="100" text-anchor="middle" font-size="10" fill="white" font-weight="bold">Conv</text>
  <text x="405" y="113" text-anchor="middle" font-size="8" fill="#D6EAF8">3x3</text>
  <text x="405" y="170" text-anchor="middle" class="dim">512 filters</text>
  <line x1="435" y1="107" x2="450" y2="107" class="arrow" marker-end="url(#arr)"/>
  <rect x="455" y="82" width="55" height="46" rx="5" fill="url(#poolGrad)"/>
  <text x="482" y="102" text-anchor="middle" font-size="9" fill="white" font-weight="bold">Global</text>
  <text x="482" y="115" text-anchor="middle" font-size="9" fill="white" font-weight="bold">Avg Pool</text>
  <text x="482" y="145" text-anchor="middle" class="dim">1792-dim</text>
  <line x1="515" y1="107" x2="535" y2="107" class="arrow" marker-end="url(#arr)"/>
  <rect x="540" y="87" width="40" height="36" rx="4" fill="#F5B7B1" stroke="#E74C3C" stroke-width="1"/>
  <text x="560" y="109" text-anchor="middle" font-size="9" fill="#922B21" font-weight="bold">Drop</text>
  <text x="560" y="140" text-anchor="middle" class="dim">p=0.3</text>
  <line x1="585" y1="107" x2="600" y2="107" class="arrow" marker-end="url(#arr)"/>
  <rect x="605" y="45" width="75" height="30" rx="5" fill="url(#fcGrad)"/>
  <text x="642" y="65" text-anchor="middle" font-size="10" fill="white" font-weight="bold">Coarse (3)</text>
  <rect x="605" y="85" width="75" height="30" rx="5" fill="url(#fcGrad)"/>
  <text x="642" y="105" text-anchor="middle" font-size="10" fill="white" font-weight="bold">Medium (15)</text>
  <rect x="605" y="125" width="75" height="30" rx="5" fill="url(#fcGrad)"/>
  <text x="642" y="145" text-anchor="middle" font-size="10" fill="white" font-weight="bold">Fine (50)</text>
  <line x1="600" y1="107" x2="605" y2="60" stroke="#7D3C98" stroke-width="1.5"/>
  <line x1="600" y1="107" x2="605" y2="100" stroke="#7D3C98" stroke-width="1.5"/>
  <line x1="600" y1="107" x2="605" y2="140" stroke="#7D3C98" stroke-width="1.5"/>
  <line x1="680" y1="60" x2="710" y2="60" class="arrow" marker-end="url(#arr)"/>
  <line x1="680" y1="100" x2="710" y2="100" class="arrow" marker-end="url(#arr)"/>
  <line x1="680" y1="140" x2="710" y2="140" class="arrow" marker-end="url(#arr)"/>
  <text x="740" y="63" class="label">Epithelial / Immune / Stromal</text>
  <text x="740" y="103" class="label">T-Cell, B-Cell, Luminal...</text>
  <text x="740" y="143" class="label">CD4+, CD8+, M1...</text>
  <rect x="115" y="195" width="14" height="14" rx="2" fill="url(#convGrad)"/>
  <text x="135" y="207" class="sublabel">Convolution + BatchNorm + ReLU</text>
  <rect x="280" y="195" width="14" height="14" rx="2" fill="url(#poolGrad)"/>
  <text x="300" y="207" class="sublabel">Pooling</text>
  <rect x="380" y="195" width="14" height="14" rx="2" fill="url(#fcGrad)"/>
  <text x="400" y="207" class="sublabel">Fully Connected Head</text>
  <rect x="510" y="195" width="14" height="14" rx="2" fill="#F5B7B1" stroke="#E74C3C" stroke-width="1"/>
  <text x="530" y="207" class="sublabel">Dropout Regularization</text>
  <line x1="115" y1="40" x2="435" y2="40" stroke="#00526B" stroke-width="1.5"/>
  <line x1="115" y1="40" x2="115" y2="48" stroke="#00526B" stroke-width="1.5"/>
  <line x1="435" y1="40" x2="435" y2="48" stroke="#00526B" stroke-width="1.5"/>
  <text x="275" y="37" text-anchor="middle" class="sublabel" fill="#00526B" font-weight="bold">Feature Extraction (Backbone)</text>
`, 'cnn_architecture.svg', 880, 225);

// ── ViT Architecture ──
saveSvg(`
  <defs>
    <marker id="arr2" viewBox="0 0 10 6" refX="10" refY="3" markerWidth="8" markerHeight="6" orient="auto">
      <path d="M0,0 L10,3 L0,6 Z" fill="#00526B"/>
    </marker>
  </defs>
  <style>
    text { font-family: Arial, sans-serif; }
    .title { font-size: 18px; font-weight: bold; fill: #00526B; }
    .label { font-size: 11px; fill: #333; }
    .sublabel { font-size: 9px; fill: #666; }
    .dim { font-size: 9px; fill: #999; font-style: italic; }
    .arrow { stroke: #00526B; stroke-width: 2; fill: none; }
  </style>
  <text x="440" y="24" text-anchor="middle" class="title">Vision Transformer (ViT) Architecture</text>
  <rect x="15" y="75" width="60" height="60" rx="4" fill="#E8F4FD" stroke="#5DADE2" stroke-width="1.5"/>
  <line x1="35" y1="75" x2="35" y2="135" stroke="#5DADE2" stroke-width="0.8"/>
  <line x1="55" y1="75" x2="55" y2="135" stroke="#5DADE2" stroke-width="0.8"/>
  <line x1="15" y1="95" x2="75" y2="95" stroke="#5DADE2" stroke-width="0.8"/>
  <line x1="15" y1="115" x2="75" y2="115" stroke="#5DADE2" stroke-width="0.8"/>
  <text x="45" y="108" text-anchor="middle" class="label" font-weight="bold">DAPI</text>
  <text x="45" y="155" text-anchor="middle" class="dim">128x128</text>
  <line x1="80" y1="105" x2="100" y2="105" class="arrow" marker-end="url(#arr2)"/>
  <rect x="105" y="60" width="18" height="18" rx="2" fill="#AED6F1" stroke="#2E86C1" stroke-width="1"/>
  <rect x="127" y="60" width="18" height="18" rx="2" fill="#AED6F1" stroke="#2E86C1" stroke-width="1"/>
  <rect x="149" y="60" width="18" height="18" rx="2" fill="#AED6F1" stroke="#2E86C1" stroke-width="1"/>
  <rect x="105" y="82" width="18" height="18" rx="2" fill="#AED6F1" stroke="#2E86C1" stroke-width="1"/>
  <rect x="127" y="82" width="18" height="18" rx="2" fill="#AED6F1" stroke="#2E86C1" stroke-width="1"/>
  <rect x="149" y="82" width="18" height="18" rx="2" fill="#AED6F1" stroke="#2E86C1" stroke-width="1"/>
  <rect x="105" y="104" width="18" height="18" rx="2" fill="#AED6F1" stroke="#2E86C1" stroke-width="1"/>
  <rect x="127" y="104" width="18" height="18" rx="2" fill="#AED6F1" stroke="#2E86C1" stroke-width="1"/>
  <text x="149" y="119" font-size="14" fill="#2E86C1" font-weight="bold">...</text>
  <text x="138" y="42" text-anchor="middle" class="label" font-weight="bold">Split into</text>
  <text x="138" y="53" text-anchor="middle" class="label" font-weight="bold">Patches</text>
  <text x="138" y="140" text-anchor="middle" class="dim">16x16 px each</text>
  <text x="138" y="155" text-anchor="middle" class="dim">N = 64 patches</text>
  <line x1="172" y1="90" x2="192" y2="90" class="arrow" marker-end="url(#arr2)"/>
  <rect x="197" y="55" width="65" height="40" rx="5" fill="#F9E79F" stroke="#F1C40F" stroke-width="1.5"/>
  <text x="229" y="73" text-anchor="middle" font-size="9" fill="#7D6608" font-weight="bold">Linear</text>
  <text x="229" y="85" text-anchor="middle" font-size="9" fill="#7D6608" font-weight="bold">Projection</text>
  <rect x="197" y="105" width="65" height="30" rx="5" fill="#FAD7A0" stroke="#E67E22" stroke-width="1.5"/>
  <text x="229" y="124" text-anchor="middle" font-size="9" fill="#784212" font-weight="bold">+ Pos Embed</text>
  <rect x="197" y="145" width="65" height="22" rx="4" fill="#D5F5E3" stroke="#27AE60" stroke-width="1.5"/>
  <text x="229" y="160" text-anchor="middle" font-size="9" fill="#1E8449" font-weight="bold">[CLS] token</text>
  <text x="229" y="185" text-anchor="middle" class="dim">768-dim vectors</text>
  <line x1="267" y1="105" x2="290" y2="105" class="arrow" marker-end="url(#arr2)"/>
  <rect x="295" y="38" width="180" height="145" rx="8" fill="#F4ECF7" stroke="#8E44AD" stroke-width="2"/>
  <text x="385" y="58" text-anchor="middle" font-size="11" fill="#6C3483" font-weight="bold">Transformer Encoder Block (x12)</text>
  <rect x="310" y="68" width="150" height="32" rx="5" fill="#D2B4DE" stroke="#8E44AD" stroke-width="1"/>
  <text x="385" y="88" text-anchor="middle" font-size="10" fill="#4A235A" font-weight="bold">Multi-Head Self-Attention</text>
  <rect x="340" y="106" width="90" height="20" rx="3" fill="#EBDEF0" stroke="#8E44AD" stroke-width="0.8"/>
  <text x="385" y="120" text-anchor="middle" font-size="8" fill="#6C3483">Add &amp; LayerNorm</text>
  <rect x="310" y="132" width="150" height="28" rx="5" fill="#D2B4DE" stroke="#8E44AD" stroke-width="1"/>
  <text x="385" y="150" text-anchor="middle" font-size="10" fill="#4A235A" font-weight="bold">Feed-Forward MLP</text>
  <rect x="340" y="165" width="90" height="16" rx="3" fill="#EBDEF0" stroke="#8E44AD" stroke-width="0.8"/>
  <text x="385" y="177" text-anchor="middle" font-size="8" fill="#6C3483">Add &amp; LayerNorm</text>
  <path d="M305,84 L298,84 L298,116 L340,116" stroke="#8E44AD" stroke-width="1" fill="none" stroke-dasharray="3,2"/>
  <path d="M305,146 L298,146 L298,173 L340,173" stroke="#8E44AD" stroke-width="1" fill="none" stroke-dasharray="3,2"/>
  <line x1="480" y1="105" x2="505" y2="105" class="arrow" marker-end="url(#arr2)"/>
  <rect x="510" y="80" width="60" height="50" rx="5" fill="#D5F5E3" stroke="#27AE60" stroke-width="1.5"/>
  <text x="540" y="100" text-anchor="middle" font-size="9" fill="#1E8449" font-weight="bold">[CLS]</text>
  <text x="540" y="115" text-anchor="middle" font-size="9" fill="#1E8449" font-weight="bold">Output</text>
  <text x="540" y="150" text-anchor="middle" class="dim">768-dim</text>
  <line x1="575" y1="105" x2="600" y2="105" class="arrow" marker-end="url(#arr2)"/>
  <rect x="605" y="90" width="35" height="30" rx="4" fill="#F5B7B1" stroke="#E74C3C" stroke-width="1"/>
  <text x="622" y="109" text-anchor="middle" font-size="9" fill="#922B21" font-weight="bold">Drop</text>
  <line x1="645" y1="105" x2="660" y2="105" class="arrow" marker-end="url(#arr2)"/>
  <rect x="665" y="50" width="75" height="28" rx="5" fill="#AF7AC5"/>
  <text x="702" y="68" text-anchor="middle" font-size="10" fill="white" font-weight="bold">Coarse (3)</text>
  <rect x="665" y="88" width="75" height="28" rx="5" fill="#AF7AC5"/>
  <text x="702" y="106" text-anchor="middle" font-size="10" fill="white" font-weight="bold">Medium (15)</text>
  <rect x="665" y="126" width="75" height="28" rx="5" fill="#AF7AC5"/>
  <text x="702" y="144" text-anchor="middle" font-size="10" fill="white" font-weight="bold">Fine (50)</text>
  <line x1="660" y1="105" x2="665" y2="64" stroke="#7D3C98" stroke-width="1.5"/>
  <line x1="660" y1="105" x2="665" y2="102" stroke="#7D3C98" stroke-width="1.5"/>
  <line x1="660" y1="105" x2="665" y2="140" stroke="#7D3C98" stroke-width="1.5"/>
  <line x1="740" y1="64" x2="770" y2="64" class="arrow" marker-end="url(#arr2)"/>
  <line x1="740" y1="102" x2="770" y2="102" class="arrow" marker-end="url(#arr2)"/>
  <line x1="740" y1="140" x2="770" y2="140" class="arrow" marker-end="url(#arr2)"/>
  <text x="780" y="67" class="label">Broad</text>
  <text x="780" y="105" class="label">Medium</text>
  <text x="780" y="143" class="label">Fine</text>
  <rect x="105" y="200" width="14" height="14" rx="2" fill="#AED6F1" stroke="#2E86C1" stroke-width="1"/>
  <text x="125" y="212" class="sublabel">Image Patches</text>
  <rect x="220" y="200" width="14" height="14" rx="2" fill="#F9E79F" stroke="#F1C40F" stroke-width="1"/>
  <text x="240" y="212" class="sublabel">Embedding</text>
  <rect x="320" y="200" width="14" height="14" rx="2" fill="#D2B4DE" stroke="#8E44AD" stroke-width="1"/>
  <text x="340" y="212" class="sublabel">Transformer Layers</text>
  <rect x="460" y="200" width="14" height="14" rx="2" fill="#D5F5E3" stroke="#27AE60" stroke-width="1"/>
  <text x="480" y="212" class="sublabel">[CLS] Token</text>
  <rect x="560" y="200" width="14" height="14" rx="2" fill="#AF7AC5"/>
  <text x="580" y="212" class="sublabel">Classification Heads</text>
`, 'vit_architecture.svg', 880, 230);

console.log('All SVGs exported to figures-svg/');
console.log(fs.readdirSync(OUT).join('\n'));
