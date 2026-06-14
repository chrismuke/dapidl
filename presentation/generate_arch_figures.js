const sharp = require('sharp');
const path = require('path');
const FIGS = path.join(__dirname, 'figures');

const SCALE = 3; // 3x resolution for high-DPI output

async function createSvgPng(svgContent, filename, width, height) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width * SCALE}" height="${height * SCALE}" viewBox="0 0 ${width} ${height}">${svgContent}</svg>`;
  await sharp(Buffer.from(svg)).png().toFile(path.join(FIGS, filename));
}

async function main() {

  // ━━━ CNN Architecture (EfficientNetV2-S style) ━━━
  await createSvgPng(`
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

    <!-- ── Input ── -->
    <rect x="15" y="70" width="55" height="55" rx="4" fill="#E8F4FD" stroke="#5DADE2" stroke-width="1.5"/>
    <rect x="20" y="75" width="55" height="55" rx="4" fill="#D4EFFC" stroke="#5DADE2" stroke-width="1.5"/>
    <rect x="25" y="80" width="55" height="55" rx="4" fill="#BEE6F9" stroke="#5DADE2" stroke-width="1.5"/>
    <text x="52" y="115" text-anchor="middle" class="label" font-weight="bold">Input</text>
    <text x="52" y="155" text-anchor="middle" class="dim">128x128x3</text>

    <line x1="85" y1="107" x2="110" y2="107" class="arrow" marker-end="url(#arr)"/>

    <!-- ── Conv Block 1 ── -->
    <rect x="115" y="55" width="50" height="100" rx="5" fill="url(#convGrad)"/>
    <text x="140" y="100" text-anchor="middle" font-size="10" fill="white" font-weight="bold">Conv</text>
    <text x="140" y="113" text-anchor="middle" font-size="8" fill="#D6EAF8">3x3</text>
    <text x="140" y="170" text-anchor="middle" class="dim">64 filters</text>

    <!-- ── Pool 1 ── -->
    <rect x="170" y="75" width="25" height="60" rx="3" fill="url(#poolGrad)"/>
    <text x="182" y="110" text-anchor="middle" font-size="7" fill="white" font-weight="bold">Pool</text>

    <line x1="200" y1="107" x2="215" y2="107" class="arrow" marker-end="url(#arr)"/>

    <!-- ── Conv Block 2 ── -->
    <rect x="220" y="60" width="50" height="90" rx="5" fill="url(#convGrad)"/>
    <text x="245" y="100" text-anchor="middle" font-size="10" fill="white" font-weight="bold">Conv</text>
    <text x="245" y="113" text-anchor="middle" font-size="8" fill="#D6EAF8">3x3</text>
    <text x="245" y="170" text-anchor="middle" class="dim">128 filters</text>

    <!-- ── Pool 2 ── -->
    <rect x="275" y="75" width="25" height="60" rx="3" fill="url(#poolGrad)"/>
    <text x="287" y="110" text-anchor="middle" font-size="7" fill="white" font-weight="bold">Pool</text>

    <line x1="305" y1="107" x2="320" y2="107" class="arrow" marker-end="url(#arr)"/>

    <!-- ── Conv Block 3 ── -->
    <rect x="325" y="65" width="50" height="80" rx="5" fill="url(#convGrad)"/>
    <text x="350" y="100" text-anchor="middle" font-size="10" fill="white" font-weight="bold">Conv</text>
    <text x="350" y="113" text-anchor="middle" font-size="8" fill="#D6EAF8">3x3</text>
    <text x="350" y="170" text-anchor="middle" class="dim">256 filters</text>

    <!-- ── Conv Block 4 (deeper) ── -->
    <rect x="380" y="70" width="50" height="70" rx="5" fill="url(#convGrad)" opacity="0.85"/>
    <text x="405" y="100" text-anchor="middle" font-size="10" fill="white" font-weight="bold">Conv</text>
    <text x="405" y="113" text-anchor="middle" font-size="8" fill="#D6EAF8">3x3</text>
    <text x="405" y="170" text-anchor="middle" class="dim">512 filters</text>

    <line x1="435" y1="107" x2="450" y2="107" class="arrow" marker-end="url(#arr)"/>

    <!-- ── Global Average Pooling ── -->
    <rect x="455" y="82" width="55" height="46" rx="5" fill="url(#poolGrad)"/>
    <text x="482" y="102" text-anchor="middle" font-size="9" fill="white" font-weight="bold">Global</text>
    <text x="482" y="115" text-anchor="middle" font-size="9" fill="white" font-weight="bold">Avg Pool</text>
    <text x="482" y="145" text-anchor="middle" class="dim">1792-dim</text>

    <line x1="515" y1="107" x2="535" y2="107" class="arrow" marker-end="url(#arr)"/>

    <!-- ── Dropout ── -->
    <rect x="540" y="87" width="40" height="36" rx="4" fill="#F5B7B1" stroke="#E74C3C" stroke-width="1"/>
    <text x="560" y="109" text-anchor="middle" font-size="9" fill="#922B21" font-weight="bold">Drop</text>
    <text x="560" y="140" text-anchor="middle" class="dim">p=0.3</text>

    <line x1="585" y1="107" x2="600" y2="107" class="arrow" marker-end="url(#arr)"/>

    <!-- ── Classification Heads ── -->
    <rect x="605" y="45" width="75" height="30" rx="5" fill="url(#fcGrad)"/>
    <text x="642" y="65" text-anchor="middle" font-size="10" fill="white" font-weight="bold">Coarse (3)</text>

    <rect x="605" y="85" width="75" height="30" rx="5" fill="url(#fcGrad)"/>
    <text x="642" y="105" text-anchor="middle" font-size="10" fill="white" font-weight="bold">Medium (15)</text>

    <rect x="605" y="125" width="75" height="30" rx="5" fill="url(#fcGrad)"/>
    <text x="642" y="145" text-anchor="middle" font-size="10" fill="white" font-weight="bold">Fine (50)</text>

    <!-- Fan-out lines to heads -->
    <line x1="600" y1="107" x2="605" y2="60" stroke="#7D3C98" stroke-width="1.5"/>
    <line x1="600" y1="107" x2="605" y2="100" stroke="#7D3C98" stroke-width="1.5"/>
    <line x1="600" y1="107" x2="605" y2="140" stroke="#7D3C98" stroke-width="1.5"/>

    <!-- Output arrows -->
    <line x1="680" y1="60" x2="710" y2="60" class="arrow" marker-end="url(#arr)"/>
    <line x1="680" y1="100" x2="710" y2="100" class="arrow" marker-end="url(#arr)"/>
    <line x1="680" y1="140" x2="710" y2="140" class="arrow" marker-end="url(#arr)"/>

    <text x="740" y="63" class="label">Epithelial / Immune / Stromal</text>
    <text x="740" y="103" class="label">T-Cell, B-Cell, Luminal...</text>
    <text x="740" y="143" class="label">CD4+, CD8+, M1...</text>

    <!-- ── Legend ── -->
    <rect x="115" y="195" width="14" height="14" rx="2" fill="url(#convGrad)"/>
    <text x="135" y="207" class="sublabel">Convolution + BatchNorm + ReLU</text>
    <rect x="280" y="195" width="14" height="14" rx="2" fill="url(#poolGrad)"/>
    <text x="300" y="207" class="sublabel">Pooling</text>
    <rect x="380" y="195" width="14" height="14" rx="2" fill="url(#fcGrad)"/>
    <text x="400" y="207" class="sublabel">Fully Connected Head</text>
    <rect x="510" y="195" width="14" height="14" rx="2" fill="#F5B7B1" stroke="#E74C3C" stroke-width="1"/>
    <text x="530" y="207" class="sublabel">Dropout Regularization</text>

    <!-- Bracket: Feature Extraction -->
    <line x1="115" y1="40" x2="435" y2="40" stroke="#00526B" stroke-width="1.5"/>
    <line x1="115" y1="40" x2="115" y2="48" stroke="#00526B" stroke-width="1.5"/>
    <line x1="435" y1="40" x2="435" y2="48" stroke="#00526B" stroke-width="1.5"/>
    <text x="275" y="37" text-anchor="middle" class="sublabel" fill="#00526B" font-weight="bold">Feature Extraction (Backbone)</text>

  `, 'cnn_architecture.png', 880, 225);


  // ━━━ Vision Transformer (ViT) Architecture ━━━
  await createSvgPng(`
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

    <!-- ── Input Image ── -->
    <rect x="15" y="75" width="60" height="60" rx="4" fill="#E8F4FD" stroke="#5DADE2" stroke-width="1.5"/>
    <!-- Grid lines to show patches -->
    <line x1="35" y1="75" x2="35" y2="135" stroke="#5DADE2" stroke-width="0.8"/>
    <line x1="55" y1="75" x2="55" y2="135" stroke="#5DADE2" stroke-width="0.8"/>
    <line x1="15" y1="95" x2="75" y2="95" stroke="#5DADE2" stroke-width="0.8"/>
    <line x1="15" y1="115" x2="75" y2="115" stroke="#5DADE2" stroke-width="0.8"/>
    <text x="45" y="108" text-anchor="middle" class="label" font-weight="bold">DAPI</text>
    <text x="45" y="155" text-anchor="middle" class="dim">128x128</text>

    <line x1="80" y1="105" x2="100" y2="105" class="arrow" marker-end="url(#arr2)"/>

    <!-- ── Patch Splitting ── -->
    <!-- Show individual patches -->
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

    <!-- ── Linear Projection + Positional Embedding ── -->
    <rect x="197" y="55" width="65" height="40" rx="5" fill="#F9E79F" stroke="#F1C40F" stroke-width="1.5"/>
    <text x="229" y="73" text-anchor="middle" font-size="9" fill="#7D6608" font-weight="bold">Linear</text>
    <text x="229" y="85" text-anchor="middle" font-size="9" fill="#7D6608" font-weight="bold">Projection</text>

    <rect x="197" y="105" width="65" height="30" rx="5" fill="#FAD7A0" stroke="#E67E22" stroke-width="1.5"/>
    <text x="229" y="124" text-anchor="middle" font-size="9" fill="#784212" font-weight="bold">+ Pos Embed</text>

    <!-- [CLS] token -->
    <rect x="197" y="145" width="65" height="22" rx="4" fill="#D5F5E3" stroke="#27AE60" stroke-width="1.5"/>
    <text x="229" y="160" text-anchor="middle" font-size="9" fill="#1E8449" font-weight="bold">[CLS] token</text>

    <text x="229" y="185" text-anchor="middle" class="dim">768-dim vectors</text>

    <line x1="267" y1="105" x2="290" y2="105" class="arrow" marker-end="url(#arr2)"/>

    <!-- ── Transformer Encoder Block (repeated) ── -->
    <rect x="295" y="38" width="180" height="145" rx="8" fill="#F4ECF7" stroke="#8E44AD" stroke-width="2"/>
    <text x="385" y="58" text-anchor="middle" font-size="11" fill="#6C3483" font-weight="bold">Transformer Encoder Block (x12)</text>

    <!-- Multi-Head Self-Attention -->
    <rect x="310" y="68" width="150" height="32" rx="5" fill="#D2B4DE" stroke="#8E44AD" stroke-width="1"/>
    <text x="385" y="88" text-anchor="middle" font-size="10" fill="#4A235A" font-weight="bold">Multi-Head Self-Attention</text>

    <!-- Add & Norm 1 -->
    <rect x="340" y="106" width="90" height="20" rx="3" fill="#EBDEF0" stroke="#8E44AD" stroke-width="0.8"/>
    <text x="385" y="120" text-anchor="middle" font-size="8" fill="#6C3483">Add &amp; LayerNorm</text>

    <!-- MLP / Feed-Forward -->
    <rect x="310" y="132" width="150" height="28" rx="5" fill="#D2B4DE" stroke="#8E44AD" stroke-width="1"/>
    <text x="385" y="150" text-anchor="middle" font-size="10" fill="#4A235A" font-weight="bold">Feed-Forward MLP</text>

    <!-- Add & Norm 2 -->
    <rect x="340" y="165" width="90" height="16" rx="3" fill="#EBDEF0" stroke="#8E44AD" stroke-width="0.8"/>
    <text x="385" y="177" text-anchor="middle" font-size="8" fill="#6C3483">Add &amp; LayerNorm</text>

    <!-- Skip connection arrows -->
    <path d="M305,84 L298,84 L298,116 L340,116" stroke="#8E44AD" stroke-width="1" fill="none" stroke-dasharray="3,2"/>
    <path d="M305,146 L298,146 L298,173 L340,173" stroke="#8E44AD" stroke-width="1" fill="none" stroke-dasharray="3,2"/>

    <line x1="480" y1="105" x2="505" y2="105" class="arrow" marker-end="url(#arr2)"/>

    <!-- ── [CLS] Output → Classification ── -->
    <rect x="510" y="80" width="60" height="50" rx="5" fill="#D5F5E3" stroke="#27AE60" stroke-width="1.5"/>
    <text x="540" y="100" text-anchor="middle" font-size="9" fill="#1E8449" font-weight="bold">[CLS]</text>
    <text x="540" y="115" text-anchor="middle" font-size="9" fill="#1E8449" font-weight="bold">Output</text>
    <text x="540" y="150" text-anchor="middle" class="dim">768-dim</text>

    <line x1="575" y1="105" x2="600" y2="105" class="arrow" marker-end="url(#arr2)"/>

    <!-- ── Dropout ── -->
    <rect x="605" y="90" width="35" height="30" rx="4" fill="#F5B7B1" stroke="#E74C3C" stroke-width="1"/>
    <text x="622" y="109" text-anchor="middle" font-size="9" fill="#922B21" font-weight="bold">Drop</text>

    <line x1="645" y1="105" x2="660" y2="105" class="arrow" marker-end="url(#arr2)"/>

    <!-- ── Classification Heads ── -->
    <rect x="665" y="50" width="75" height="28" rx="5" fill="#AF7AC5"/>
    <text x="702" y="68" text-anchor="middle" font-size="10" fill="white" font-weight="bold">Coarse (3)</text>
    <rect x="665" y="88" width="75" height="28" rx="5" fill="#AF7AC5"/>
    <text x="702" y="106" text-anchor="middle" font-size="10" fill="white" font-weight="bold">Medium (15)</text>
    <rect x="665" y="126" width="75" height="28" rx="5" fill="#AF7AC5"/>
    <text x="702" y="144" text-anchor="middle" font-size="10" fill="white" font-weight="bold">Fine (50)</text>

    <!-- Fan lines -->
    <line x1="660" y1="105" x2="665" y2="64" stroke="#7D3C98" stroke-width="1.5"/>
    <line x1="660" y1="105" x2="665" y2="102" stroke="#7D3C98" stroke-width="1.5"/>
    <line x1="660" y1="105" x2="665" y2="140" stroke="#7D3C98" stroke-width="1.5"/>

    <!-- Output labels -->
    <line x1="740" y1="64" x2="770" y2="64" class="arrow" marker-end="url(#arr2)"/>
    <line x1="740" y1="102" x2="770" y2="102" class="arrow" marker-end="url(#arr2)"/>
    <line x1="740" y1="140" x2="770" y2="140" class="arrow" marker-end="url(#arr2)"/>
    <text x="780" y="67" class="label">Broad</text>
    <text x="780" y="105" class="label">Medium</text>
    <text x="780" y="143" class="label">Fine</text>

    <!-- ── Legend ── -->
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

  `, 'vit_architecture.png', 880, 230);

  console.log('CNN and ViT architecture figures generated!');
}

main().catch(console.error);
