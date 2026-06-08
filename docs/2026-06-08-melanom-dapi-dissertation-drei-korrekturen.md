# Drei Korrekturen zur Stärkung der Melanom-DAPI-Dissertation

**Betreff:** Melanom-DAPI-Dissertationsskizze — *„Charakterisierung von Melanomen durch Deep-Learning-basierte Zellkern-Segmentierung und Klassifikation"*
**Datum:** 08.06.2026
**Status:** konstruktive Begutachtung vor Implementierungsbeginn

---

## Warum dieses Dokument

Der Kern des Projekts ist tragfähig und lohnenswert: DAPI-basierte Zellkern-Segmentierung und DAPI-basierte Zelltyp-Klassifikation am Melanom, validiert gegen manuelle immunhistochemische Quantifizierung, ist ein verteidigungsfähiges und publizierbares Dissertationsthema. Das Deep-Learning-Modell funktioniert für die grobe Zellidentität bereits, die hauseigenen Maus-Melanom-DAPI-Aufnahmen sind eine seltene und wertvolle Ressource (kein öffentlicher räumlich-transkriptomischer Maus-Melanom-Datensatz stellt herunterladbare DAPI-Bilder bereit), und das Validierungsdesign ist realistisch.

Drei Aussagen der aktuellen Projektskizze sind jedoch wissenschaftlich angreifbar und würden in einer Disputation hinterfragt. Alle drei sind **korrigierbar, ohne das Projekt zu schwächen** — die Korrekturen machen es im Gegenteil stärker und leichter verteidigbar. Jede Korrektur nennt das Problem, die Evidenz und die konkrete Abhilfe.

> Hinweis zu den Quellen: Die folgenden PMIDs/DOIs stammen aus einer unabhängigen Literaturrecherche. **Bitte jede Quelle vor dem Zitieren in der Dissertation selbst prüfen** — das ist ohnehin gute wissenschaftliche Praxis.

| Nr. | Aussage in der Skizze | Das Problem | Die Korrektur |
|---|---|---|---|
| 1 | DAPI zeige „subtile perinukleäre mRNA-Signale" | Kategorienfehler — DAPI färbt dsDNA, keine mRNA | mRNA-Aussage streichen; Mechanismus über **Chromatin-/Kernmorphologie** begründen |
| 2 | „Aktivierte" (CD8⁺CD137⁺, GzmB) vs. „erschöpfte" (CD8⁺PD‑1⁺, TIM‑3⁺) als zwei Klassen unterscheiden | Die Marker-Definitionen sind biologisch unhaltbar; die Zustände sind ein Kontinuum | **Jeden Marker einzeln (Multilabel)** vorhersagen; aktiviert/erschöpft nur explorativ, nicht als Primärziel |
| 3 | Das Modell klassifiziere diese Zustände aus DAPI mit klinischer Präzision | Zustand aus DAPI ist unbelegt; belegt ist die **Identität** aus DAPI | **Ehrliche Obergrenze** setzen; immunologische **Zusammensetzung** zum primären Endpunkt machen |

---

## Korrektur 1 — „Perinukleäre mRNA-Signale aus DAPI" ist ein Kategorienfehler

**Was die Skizze sagt.** Dass die DAPI-Färbung „kaum interpretierbare perinukleäre mRNA-Signale" sichtbar mache und das Modell diese ausnutze.

**Warum das ein Problem ist.** DAPI (4′,6-Diamidin-2-phenylindol) bindet an die **AT-reiche kleine Furche doppelsträngiger DNA** und fluoresziert dort stark (~460 nm). An RNA bindet es nur schwach und unspezifisch, mit rotverschobener, deutlich schwächerer Emission, die von Standard-DAPI-Filtersätzen aktiv zurückgewiesen wird — und es kann mRNA nicht von rRNA unterscheiden oder ein Transkript identifizieren (Kapuściński & Szer 1979; Kapuściński 1995). Der Nachweis von **mRNA** in situ erfordert sequenzspezifische RNA-FISH-/sondenbasierte Verfahren (z. B. Einzelmolekül-FISH), wobei DAPI nur als separate Kerngegenfärbung dient. Ein etwaiges echtes perinukleäres DAPI-Signal wäre **mitochondriale oder zytoplasmatische dsDNA**, keine mRNA.

Gutachter:innen oder Prüfer:innen erkennen dies sofort, und es untergräbt die Glaubwürdigkeit des Methodenteils.

**Die Korrektur.** Die mRNA-Aussage vollständig streichen. Der verteidigungsfähige — und weiterhin neuartige — Mechanismus ist die **Kernmorphologie und Chromatinorganisation**: Kerngröße und -form, Heterochromatin-Kondensation/-Verklumpung, Prominenz der Nukleoli und Integrität der Kernhülle. Das ist ein reales, bildgebend erfassbares Signal (siehe Korrektur 3) und genau das, was das Modell tatsächlich lernt.

---

## Korrektur 2 — Die Aktiviert/Erschöpft-Dichotomie ist in dieser Form biologisch unhaltbar

**Was die Skizze sagt.** Zwei sauber getrennte CD8-T-Zell-Klassen: **aktiviert** = CD8⁺CD137⁺ / Granzym B; **erschöpft** = CD8⁺PD‑1⁺ / TIM‑3⁺.

**Warum das ein Problem ist.** Die Marker-Logik trägt nicht:

- **PD‑1 wird durch normale T-Zell-Aktivierung induziert**, nicht nur durch Erschöpfung (PMID 18802087). „CD8⁺PD‑1⁺ = erschöpft" fehletikettiert damit kürzlich aktivierte Effektorzellen als erschöpft.
- **TIM‑3 ist für Erschöpfung weder notwendig noch hinreichend** und markiert auch Effektor-Aktivierung (PMID 29463725).
- **Granzym B und TIM‑3 treten gemeinsam in proliferierenden, transitionellen Populationen** auf, die *innerhalb* der Erschöpfungstrajektorie liegen — die Marker-Sets „aktiviert" und „erschöpft" überlappen also auf denselben Zellen (PMID 31810882).
- Speziell im Melanom bilden **CD8-T-Zell-Zustände ein Kontinuum**, keine zwei diskreten Klassen (PMID 30595452).

Ein Klassifikator, der auf biologisch falschen Labels trainiert wird, lernt das falsche Ziel — und dieser Fehler ist nachgelagert nicht mehr korrigierbar.

**Die Korrektur.** Kein Zwei-Klassen-Label „aktiviert vs. erschöpft" vorhersagen. Stattdessen:

1. Jeden Marker — **CD137, Granzym B, PD‑1, TIM‑3** — als **separaten binären (Multilabel-)Endpunkt** behandeln. Das trifft keine Annahme darüber, wie sich Marker zu Zuständen kombinieren.
2. Die Marker-Leistung einzeln berichten und *erst danach*, als **sekundäre, explorative** Analyse, Marker-Koinzidenzmuster (z. B. PD‑1⁺TIM‑3⁺ vs. CD137⁺GzmB⁺) als Kandidaten-Signaturen für Zustände untersuchen.
3. „Aktivierung" und „Erschöpfung" als **interpretative Hypothesen über Marker-Kombinationen** rahmen, niemals als die Ground-Truth-Klassen, auf die das Modell trainiert wird.

Das ist ehrlicher, flexibler und — entscheidend — erlaubt weiterhin die Aktivierungs-/Erschöpfungs-Erzählung, nur ohne die Dissertation auf eine Definition zu stützen, die Prüfer:innen zurückweisen werden.

---

## Korrektur 3 — Zell*identität* aus DAPI ist belegt; funktioneller *Zustand* aus DAPI ist unbelegt

**Was die Skizze sagt.** Dass das etablierte Modell diese Immunzustände aus DAPI „mit klinisch relevanter Präzision und Robustheit" auflöst.

**Warum das ein Problem ist — und wo die Skizze recht hat.** Es gibt starke Präzedenz, dass eine **DNA-Färbung allein tiefe biologische Information trägt**:

- DAPI-Kernmorphologie sagt **zelluläre Seneszenz** mit bis zu ~95 % Genauigkeit über Zelltypen und Spezies hinweg vorher (Heckenbach et al. 2022, *Nature Aging*, 10.1038/s43587-022-00263-3).
- Ein 3D-DAPI-only-CNN klassifiziert **Nierenzelltypen** mit ~80 % balancierter Genauigkeit (Woloshuk et al. 2021, *Cytometry A*, 10.1002/cyto.a.24274).
- T-Zell-**Aktivierung** führt zu bildgebend erfassbarer Chromatin-Dekondensation und Veränderungen der Kernhülle (Xu et al. 2024, *Commun Biol*, 10.1038/s42003-024-06479-w; Bediaga et al. 2021, *Sci Rep*), und Immunzell-Morphologie bildet den transkriptionellen Zustand ab (Severin et al. 2021, *Sci Adv*, 10.1126/sciadv.abf6692). Erschöpfung geht mit genomweiter Chromatin-Umstrukturierung einher (Sen et al. 2016; Pauken et al. 2016, *Science*).

**Zellidentität aus Kernen ist also literaturbelegt**, und es *gibt* eine reale morphologische Grundlage für den Aktivierungszustand. **Aber:**

- **Keine publizierte Arbeit zeigt, dass DAPI *erschöpfte* von *aktivierten* CD8-T-Zellen unterscheidet.** Diese spezifische Aussage ist eine Hypothese, kein etabliertes Ergebnis.
- Die stützende Evidenz stammt aus hochauflösender Konfokal-/Superresolution-Mikroskopie. Bei räumlich-transkriptomischer Auflösung (~0,2 µm/Pixel) ist ein 5–7 µm großer Lymphozytenkern nur ~10–25 Pixel breit — subtile innere Chromatin-Topologie ist weitgehend nicht auflösbar, und scheinbare „Textur" in dichtem Gewebe ist oft physische Verdrängung, nicht Chromatinzustand.

**Die Korrektur — ehrliche Obergrenze und Endpunkt-Hierarchie.**

- **Primärer Endpunkt:** immunologische **Zusammensetzung** (z. B. CD8 vs. Nicht-CD8; Immun-/Tumor-/Stroma-Anteile pro Field/ROI). Robust und verteidigungsfähig.
- **Sekundärer Endpunkt:** Marker-Positivität pro Marker (Korrektur 2).
- **Explorativ:** Aktivierungs-/Erschöpfungs-Signaturen.
- **Quantitative Erwartung:** unter ehrlicher, batch-unabhängiger Testung ist mit einer **AUROC ≈ 0,65–0,70** pro Marker zu rechnen (Makro-F1 ≈ 0,55–0,62). **Jede AUROC über ~0,75 als Warnsignal für Confounding behandeln** (z. B. das Modell erkennt die Maus, den Objektträger oder den Färbe-Batch statt der T-Zell-Biologie), bis sie auf einem zurückgehaltenen Batch unabhängig repliziert ist.

---

## Der angepasste Arbeitstitel

> **„Bewertung DAPI-abgeleiteter nuklearer und räumlicher Morphologie als Surrogat für marker-definierte CD8-Phänotypen und immunologische Zusammensetzung im Maus-Melanom."**

Das sagt genau, was die Studie verteidigen kann: Sie misst *DAPI-Kern-/Raummorphologie* gegen *marker-definierte* Phänotypen und *Zusammensetzung* — ohne Überzeichnung bezüglich des Lesens von mRNA oder der Diagnose diskreter Erschöpfungszustände.

---

## Was vollständig tragfähig bleibt (die verteidigungsfähige Dissertation)

- **DAPI-Zellkern-Segmentierung** am Melanom, validiert gegen manuelle Konturen (Detektions-F1, PQ/AJI, Dice). Das nötige QC-/Segmentierungs-Werkzeug existiert bereits und ist produktionsreif.
- **Grobe DAPI-basierte Zelltypisierung** (Immun/Tumor/Stroma, CD8 vs. Nicht-CD8), validiert gegen die manuellen immunhistochemischen Zählungen.
- Eine rigorose **Domänentransfer-Analyse** (Mensch→Maus, Mamma→Melanom) — selbst ein publizierbarer Beitrag.
- Eine prinzipielle **Obergrenze** dessen, was DAPI über den funktionellen T-Zell-Zustand aussagen kann — ein *rigoroses negatives oder moderat positives Ergebnis, das eine Disputation übersteht*, was eine geleakte, konfundierte AUROC von 0,90 nicht tut.

Keine der drei Korrekturen verkleinert die Dissertation. Sie verschieben die starken Aussagen von *unhaltbar* zu *genau haltbar* — und genau das bringt eine Promotion durch die Prüfung.

---

### Zu prüfende und zu zitierende Quellen

- Kapuściński J. & Szer W. (1979) *Nucleic Acids Res* — DAPI fluoresziert mit dsDNA, nicht mit RNA.
- Kapuściński J. (1995) *Biotech Histochem* 70:220–233 — DAPI als DNA-spezifische Sonde.
- PD‑1 durch T-Zell-Aktivierung induziert — PMID 18802087.
- TIM‑3 nicht erschöpfungsspezifisch — PMID 29463725.
- Granzym B⁺TIM‑3⁺ transitionelle Populationen — PMID 31810882.
- Melanom-CD8-Zustände bilden ein Kontinuum — PMID 30595452.
- Heckenbach et al. (2022) *Nature Aging* — 10.1038/s43587-022-00263-3.
- Woloshuk et al. (2021) *Cytometry A* — 10.1002/cyto.a.24274.
- Severin et al. (2021) *Science Advances* — 10.1126/sciadv.abf6692.
- Xu et al. (2024) *Communications Biology* — 10.1038/s42003-024-06479-w.
- Bediaga et al. (2021) *Scientific Reports* — 10.1038/s41598-021-93180-1.
- Sen et al. (2016) / Pauken et al. (2016) *Science* — Chromatin-Umstrukturierung bei T-Zell-Erschöpfung.
