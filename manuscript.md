---
author-meta:
- Brian Orcutt-Jahns
- Zhixin Cyrillus Tan
- Aaron S. Meyer
bibliography: []
date-meta: '2020-10-08'
header-includes: '<!--

  Manubot generated metadata rendered from header-includes-template.html.

  Suggest improvements at https://github.com/manubot/manubot/blob/master/manubot/process/header-includes-template.html

  -->

  <meta name="dc.format" content="text/html" />

  <meta name="dc.title" content="A quantitative view of strategies to engineer cell-selective ligand binding" />

  <meta name="citation_title" content="A quantitative view of strategies to engineer cell-selective ligand binding" />

  <meta property="og:title" content="A quantitative view of strategies to engineer cell-selective ligand binding" />

  <meta property="twitter:title" content="A quantitative view of strategies to engineer cell-selective ligand binding" />

  <meta name="dc.date" content="2020-10-08" />

  <meta name="citation_publication_date" content="2020-10-08" />

  <meta name="dc.language" content="en-US" />

  <meta name="citation_language" content="en-US" />

  <meta name="dc.relation.ispartof" content="Manubot" />

  <meta name="dc.publisher" content="Manubot" />

  <meta name="citation_journal_title" content="Manubot" />

  <meta name="citation_technical_report_institution" content="Manubot" />

  <meta name="citation_author" content="Brian Orcutt-Jahns" />

  <meta name="citation_author_institution" content="Department of Bioengineering, University of California, Los Angeles" />

  <meta name="citation_author_orcid" content="0000-0002-1436-1224" />

  <meta name="citation_author" content="Zhixin Cyrillus Tan" />

  <meta name="citation_author_institution" content="Bioinformatics Interdepartmental Program, University of California, Los Angeles" />

  <meta name="citation_author_orcid" content="0000-0002-5498-5509" />

  <meta name="citation_author" content="Aaron S. Meyer" />

  <meta name="citation_author_institution" content="Department of Bioengineering, University of California, Los Angeles" />

  <meta name="citation_author_institution" content="Bioinformatics Interdepartmental Program, University of California, Los Angeles" />

  <meta name="citation_author_institution" content="Jonsson Comprehensive Cancer Center, University of California, Los Angeles" />

  <meta name="citation_author_institution" content="Eli and Edythe Broad Center of Regenerative Medicine and Stem Cell Research, University of California, Los Angeles" />

  <meta name="citation_author_orcid" content="0000-0003-4513-1840" />

  <meta name="twitter:creator" content="@aarmey" />

  <link rel="canonical" href="https://meyer-lab.github.io/cell-selective-ligands/" />

  <meta property="og:url" content="https://meyer-lab.github.io/cell-selective-ligands/" />

  <meta property="twitter:url" content="https://meyer-lab.github.io/cell-selective-ligands/" />

  <meta name="citation_fulltext_html_url" content="https://meyer-lab.github.io/cell-selective-ligands/" />

  <meta name="citation_pdf_url" content="https://meyer-lab.github.io/cell-selective-ligands/manuscript.pdf" />

  <link rel="alternate" type="application/pdf" href="https://meyer-lab.github.io/cell-selective-ligands/manuscript.pdf" />

  <link rel="alternate" type="text/html" href="https://meyer-lab.github.io/cell-selective-ligands/v/fb2954db831b245fde9b9bcc315e0ed68259d103/" />

  <meta name="manubot_html_url_versioned" content="https://meyer-lab.github.io/cell-selective-ligands/v/fb2954db831b245fde9b9bcc315e0ed68259d103/" />

  <meta name="manubot_pdf_url_versioned" content="https://meyer-lab.github.io/cell-selective-ligands/v/fb2954db831b245fde9b9bcc315e0ed68259d103/manuscript.pdf" />

  <meta property="og:type" content="article" />

  <meta property="twitter:card" content="summary_large_image" />

  <link rel="icon" type="image/png" sizes="192x192" href="https://manubot.org/favicon-192x192.png" />

  <link rel="mask-icon" href="https://manubot.org/safari-pinned-tab.svg" color="#ad1457" />

  <meta name="theme-color" content="#ad1457" />

  <!-- end Manubot generated metadata -->'
keywords:
- bioengineering
- protein therapies
- systems biology
lang: en-US
manubot-clear-requests-cache: false
manubot-output-bibliography: output/references.json
manubot-output-citekeys: output/citations.tsv
manubot-requests-cache-path: cache/requests-cache
title: A quantitative view of strategies to engineer cell-selective ligand binding
...



<small><em>
This manuscript
was automatically generated on October 8, 2020.
</em></small>

## Authors


+ **Brian Orcutt-Jahns**<br>
    ORCID
    [0000-0002-1436-1224](https://orcid.org/0000-0002-1436-1224)
    · Github
    [borcuttjahns](https://github.com/borcuttjahns)<br>
  <small>
     Department of Bioengineering, University of California, Los Angeles
  </small>

+ **Zhixin Cyrillus Tan**<br>
    ORCID
    [0000-0002-5498-5509](https://orcid.org/0000-0002-5498-5509)
    · Github
    [cyrillustan](https://github.com/cyrillustan)<br>
  <small>
     Bioinformatics Interdepartmental Program, University of California, Los Angeles
  </small>

+ **Aaron S. Meyer**<br>
    ORCID
    [0000-0003-4513-1840](https://orcid.org/0000-0003-4513-1840)
    · Github
    [aarmey](https://github.com/aarmey)
    · twitter
    [aarmey](https://twitter.com/aarmey)<br>
  <small>
     Department of Bioengineering, University of California, Los Angeles; Bioinformatics Interdepartmental Program, University of California, Los Angeles; Jonsson Comprehensive Cancer Center, University of California, Los Angeles; Eli and Edythe Broad Center of Regenerative Medicine and Stem Cell Research, University of California, Los Angeles
  </small>



## Abstract {.page_break_before}

A critical property of many therapies is their selective binding to specific target populations. Exceptional specificity can arise from high-affinity binding to unique cell surface targets. In many cases, however, therapeutic targets are only expressed at subtly different levels relative to off-target cells. More complex binding strategies have been developed to overcome this limitation, including multi-specific and multi-valent molecules, but create a combinatorial explosion of design possibilities. Therefore, guiding strategies for developing cell-specific binding are critical to employ these tools. Here, we extend a multi-valent binding model to multi-ligand and multi-receptor interactions. Using this model, we explore a series of mechanisms to engineer cell selectivity, including mixtures of molecules, affinity adjustments, and valency changes. Each of these strategies maximizes selectivity in distinct cases, leading to synergistic improvements when used in combination. Finally, we identify situations in which selectivity cannot be derived through passive binding alone to highlight areas in need of new developments. In total, this work uses a quantitative model to unify a comprehensive set of design guidelines for engineering cell-specific therapies.

## Summary points

- Affinity, valency, and other alterations to target cell binding provide enhanced selectivity in specific situations.
- Evidence for the effectiveness and limitations of each strategy are abundant within the drug development literature.
- Combining strategies can offer enhanced selectivity.
- A simple, multivalent ligand-receptor binding model can help to direct therapeutic engineering.



## Introduction

<!-- Targeting specific cell populations is a universal challenge in protein therapies. -->

Many drugs derive their therapeutic benefit and avoid toxicity through selective binding to specific cells within the body. However, even with a drug of very specific molecular binding, genetic and non-genetic heterogeneity can create a wide distribution of cell responses. This can result in reduced effectiveness and toxicity. For example, in cancer, resistance to anti-tumor antibodies [@doi:10.1158/1078-0432.CCR-09-1735], targeted inhibitors [@pmid:20129249], chemotherapies [@doi:10.1016/j.cell.2016.03.025], and chimeric antigen receptor T cells [@doi:10.1056/NEJMoa1407222; @doi:10.1126/scitranslmed.aaa0984] all can arise through the selection of cells within a heterogeneous population. Indeed, the immune system takes advantage of heterogeneity at the single-cell level to translate noisy inflammatory signals into robust yet sensitive responses [@PMID:24919153]. Even in the absence of heterogeneity, target cells can differ from off-target populations only by subtle changes in surface receptor expression. These challenges limit the selectivity of therapies within the body.

<!-- Need alternative strategies for engineering specificity. -->

Further improving cell-specific targeting will require new strategies of engineering specificity. Protein therapies have most extensively been engineered to target specific cell types through mutations that provide high-affinity binding to unique surface antigens [@pmid:25992859]. This strategy can enhance specificity, but only to a limited degree, particularly when target cells can only be distinguished by subtle quantitative differences in surface antigen or by combinations of markers. The limitations of single-antigen targeting have led to efforts to engineer logic and more complex programs into cellular therapies to recognize target cells more specifically [@doi:10.1126/science.aay2790; @pmid:30889382; @doi:10.1016/j.cell.2018.03.038]. However, protein and other non-cellular therapies have considerable benefits in drug access, manufacturing, and reliability; some of the same benefits have begun to be engineered into these agents.

<!-- We need computational models to make sense of this. -->
The enormous number of potential configurations make computational tools essential for designing highly selective therapies [@doi:10.1101/812131]. Here, we analyze a suite of molecular strategies for engineering cell-specific binding using a multi-valent, multi-receptor, multi-ligand model. We show that strategies including affinity, valency, binding competition, ligand mixtures, and hetero-valent complexes provide distinct and non-overlapping improvements in cell-specific targeting. With each strategy, we highlight evidence of its effect in existing therapeutic agents. Finally, we combine these strategies to target cells through combinatorial strategies. In total, our results demonstrate that binding programs can offer combinatorial targeting strategies with similar effectiveness as complex engineered cellular therapies and can be engineered using a mechanistic binding model.


## Results

### A Model System to Explore the Factors Contributing to Cell Selectivity

![**A model system for exploring the factors contributing to cell selectivity.** a) A simplified schematic of the binding model. There are two types of receptors and two types of ligand monomers that form a tetravalent complex. b) A cartoon for four cell populations expressing two different receptors at low or high amounts. c) A sample heat/contour map for the model-predicted log ligand bound given the expression of two types of receptors. d) Eight theoretical cell populations with various receptor expression profiles.](/Users/cyrillus/Desktop/manuscript/figure1.svg){#fig:model}

Here we investigate cell-specific targeting quantitatively by extending a multi-valent, multi-ligand equilibrium binding model. Virtually any therapy, including monoclonal antibodies, small-molecule inhibitors, and cytokines, can be thought of as ligands for respective receptors expressed on target cells. Ligand binding to target cells is the first step and essential for a drug's intended action; in contrast, binding to off-target cells can result in unintended effects or toxicity. Some cell populations can be distinguished by unique receptor expression, but more commonly target and off-target cells express the same collection of receptors and differ only in their amounts. In such situations, engineering drugs optimized to achieve more ligand bound to the target cells while minimizing the binding to off-target cells, is an area of ongoing research and has inspired myriad drug design strategies [@doi:10.1038/sj.bjc.6604700; @doi:10.1021/cb6003788; @doi:10.1038/s41586-018-0830-7]. In this work, we define cell population selectivity as the ratio of the number of ligands bound to target cell populations over the number of ligands bound to off-target cell populations. We will use a quantitative binding estimation for each cell population to examine these strategies.

While ligand-receptor binding events in biology are diverse, they are governed by thermodynamic properties and the law of mass action. When a reaction reaches its equilibrium state, the ratio of the concentration of monomeric ligand-receptor complex, $[LR]$, to the product of the concentrations of each, $[L][R]$, can be defined as an association constant $K_a$, and its reciprocal is defined as the dissociation constant $K_d$. The number of ligands in the solution is usually much greater than that of the receptors, so we assume binding does not deplete the ligand concentration. Binding behavior is more complex when the ligands are multivalent complexes consisting of multiple units, each of which can bind to a receptor (Fig. {@fig:model}a). During initial association, we assume that the first subunit on a ligand complex binds according to the same dynamics that govern monovalent binding. Subsequent binding events exhibit different behavior, however, due to the increased local concentration of the complex and steric effects. Here, we assume that the effective association constant for the subsequent bindings is proportional to that of the free binding, but scaled by a constant, $K_x^*$. Another consideration that must be made when modeling multivalent binding processes is whether a ligand complex is formed by a random assortment of monomer units or by engineering to produce complexes of defined uniform composition. We developed a multivalent binding model that calculates the amount of ligand bound at equilibrium taking each of these factors into account (see methods).

As a simplification, we will consider theoretical cell populations that express only two receptors capable of binding ligand (Fig. {@fig:model}b), ranging in abundance from 100 to 10,000 per cell. Figure {@fig:model}c shows the log-scaled predicted amount of binding of a monovalent ligand given the abundance of two receptors, with the concentration of ligand, $L_0$, to be $1 \mathrm{nM}$, and its dissociation constants to the two receptors to be $10 \mathrm{\mu M}$ and $100$ $\mathrm{nM}$, respectively. Because all axes are log-scaled, the number of contour lines between two points indicates the ratio of ligand binding between populations. For instance, in figure {@fig:model}c, cell populations at points 1 and 2 are on the same contour line and thus have the same amount of ligand bound; the cell populations at points 1 and 3 have multiple contour lines between and so point 3 has more ligand binding (In fact, the ratio can be read as the exponent of the contour line difference. For point 3 to point 1, the ratio is $e ^{4.6-2.3} \approx 7.4$). Alternatively, we can think of moving from one point to another as a change of expression profile for a cell population. When the amount of $R_1$ increases on a cell (a point moves rightward, e.g. from 1 to 2), the amount of binding doesn’t increase significantly. On the contrary, adding more $R_2$ (moving upward, e.g. from 1 to 3) will lead to a great boost on ligand bound. The high affinity of $R_2$ leads to a greater binding sensitivity on the expression of $R_2$ than $R_1$. This situation might correspond to some cue inducing expression of a receptor, such as interferon-induced upregulation of MHC and regulatory T cell upregulation of IL-2R⍺ [@DOI:10.1073/pnas.0812851107].

To analyze more general cases, we defined eight theoretical cell populations according to their differential expression of two receptor types ($R_1$ and $R_2$ plotted on x and y axes). As shown in Fig {@fig:model}d, they either have high ($10^4$), medium ($10^3$), or low ($10^2$) expression of $R_1$ and $R_2$. To account for cell-to-cell heterogeneity, we also defined intrapopulation variability. For instance, the expression profile of $R_1^{med} R_2^{med}$ has a wider range. We will use this binding model to examine how engineering using affinity, valency, and hetero-valent complexes improve cell-specific targeting. Although we will only consider two receptor and ligand subunit types respectively, the principles we present can generalize to more complex cases.

### Affinity Provides Selectivity Toward Cell Populations with Divergent Receptor Expression

![**Affinity provides selectivity to cell populations with divergent receptor expression.** a) Heat/contour maps of monovalent ligand binding to cell populations given the surface abundance of two receptors. Ligand dissociation constants to these receptors range from $10 \mathrm{\mu M} \sim 100 \mathrm{nM}$. Ligand concentration $L_0=1 \mathrm{nM}$. b-e) Heatmap of binding ratio of cell populations exposed to a monovalent ligand with dissociation constants to receptor 1 and 2 ranging from $10^{4} \sim 10^{2} \mathrm{nM}$, at a concentration $L_0 = 1 \mathrm{nM}$. Ligand bound ratio of (b) $R_1^{hi}R_2^{lo}$ to $R_1^{lo}R_2^{hi}$, (c) $R_1^{med}R_2^{hi}$ to $R_1^{hi}R_2^{med}$, (d) $R_1^{hi}R_2^{lo}$ to $R_1^{med}R_2^{lo}$, and (e) $R_1^{hi}R_2^{hi}$ to $R_1^{med}R_2^{med}$.](figure2.svg){#fig:affinity}

We first explored altering the affinities of a ligand to the receptors as an engineering strategy to enhance its cell population specificity. Here, we showed the binding pattern of monovalent ligands with various affinities ranging from $10000 \mathrm{nM}$ to $100 \mathrm{nM}$ to the two receptors $R_1$ and $R_2$ (Fig. {@fig:affinity}a). When a target cell population expresses a receptor uniquely higher than other populations, enhancing the affinity of the drug to this receptor is a clear strategy to increase selective binding to this population. For example, $R_1^{hi}R_2^{lo}$ only expresses a high amount of $R_1$, while $R_1^{lo}R_2^{hi}$ only expresses a high amount of $R_2$. When the affinity to $R_1$ is enhanced and the affinity to $R_2$ is reduced, the binding selectivity towards $R_1^{hi}R_2^{lo}$ increases (Fig. {@fig:affinity}b). The contour plots in Fig. {@fig:affinity}a shows this trend more intuitively: when the affinity of $R_1$ increases, which corresponds to shifting from the left subplots to the right ones, the contour lines of higher values shift inward and cluster more densely, indicating an increase in the amount of ligand bound for the populations with high $R_1$ expression, such as $R_1^{hi}R_2^{hi}$, $R_1^{hi}R_2^{med}$, and $R_1^{hi}R_2^{lo}$.

However, it is quite often that both on- and off-target cell populations express the same set of receptors and only differ in their magnitude of expression. In these cases, we found out that it is beneficial for the drug to bind tightly to the most comparatively highly expressed receptor on the target population. Some examples of this pattern are the selective binding to $R_1^{med}R_2^{hi}$ over $R_1^{hi}R_2^{med}$, and $R_1^{hi}R_2^{lo}$ over $R_1^{med}R_2^{lo}$ (Fig. {@fig:affinity}a). For these two pairs, the benefit of affinity changes is limited to the relative discrepancy in receptor amounts (Fig. {@fig:affinity}c,d). Since the more highly expressed receptors have only 10-fold difference comparing with the 100-fold difference in $R_1^{hi}R_2^{lo}$ over $R_1^{lo}R_2^{hi}$, the greatest binding ratios these two pairs can achieve are no more than 10 times instead of 50 times. When both receptors are uniformally more abundant in a target population than the off target population, such as when comparing $R_1^{hi}R_2^{hi}$ to $R_1^{med}R_2^{med}$, affinity tuning doesn't change selective binding (Fig. {@fig:affinity}d). Therefore, to use affinity changes for selectivity enhancement, it is critical to identify which receptors a cell population of interest expresses the most uniquely.

Through reading the contour plots, we can also ponder how affinity changes affect the intrapopulation heterogeneity. For example, $R_1^{med}R_2^{med}$ has relatively high variance in both $R_1$ and $R_2$ expression (Fig. @fig:affinity a). When the binding affinities to $R_1$ and $R_2$ are divergent, such as shown in the subplots in the upper left corner and bottom right corner of Figure @fig:affinity a, the ellipse representing its range of receptor abundances rides through multiple contour lines, indicating wide variation in the quantity of bound ligand. This intrapopulation variation in the amount of ligand bound, however, is not observed when the affinities to $R_1$ and $R_2$ are more balanced. When designing a therapeutic for cell populations with broader or more dynamic expression profile, such as tumor cells, the consideration in intrapopulation heterogeneity when using affinity engineering is worthwhile.

### Valency Enables Selectivity Based on Quantitative Differences in Receptor Abundance

Given the limitations of deriving selectivity from affinity changes, we next explored the effect of valency changes (Fig. {@fig:valency}). Multivalent ligand binding differs from monovalent binding in its non-linear relationship with cellular receptor density, allowing multivalent ligands to potentially selectively target cells based on receptor abundance [@doi:10.1038/srep40098]. The effects of varying ligand valency are visualized in Fig. {@fig:valency}a. Here, the amount of ligand bound by cells expressing different abundances of receptor $R_1$ and $R_2$ is quantified. The ligand only binds to receptor $R_1$ and the valency of the ligand is varied across subplots. The most notable effect of changing valency is observed as variable density of lines in the binding contour plot. For example, in the monovalent case, there are roughly the same amount of contour lines between $ R_1^{lo}R_2^{lo}$ and $R_1^{med}R_2^{med}$ and between $R_1^{med}R_2^{med}$  and $R_1^{hi}R_2^{hi}$, but in the tetravalent case, there are more contour lines between $R_1^{med}R_2^{med}$  and $R_1^{hi}R_2^{hi}$, suggesting a sharper increase in ligand bound (Fig. {@fig:valency}a).

Previous work has shown that our binding model can accurately predict the complicated multivalent binding process between IgG antibodies and Fcγ receptors [@pmid:29960887]. To further demonstrate our model's capacity to predict multivalent binding activity, we fit our model to previous experimental measurements wherein fluorescently labeled nanorings were assembled with specific numbers of binding units [@doi:10.1021/jacs.8b09198]. After fitting to determine the crosslinking coefficient and receptor affinity values, our model was able to accurately match the binding of nanorings with four or eight binding units to MCF-7 cells expressing a known abundance of receptor partners (Fig. {@fig:Csizmar}a).

Selectivity derived by changing valency is dependent on affinity. As many previous studies have suggested, lower affinity, multi-valent interactions have been shown to exhibit selectivity based on receptor abundance [@doi:10.1021/cb6003788; @doi:10.1021/jacs.8b09198]. Comparing binding between cell populations exposed to ligands of variable affinities over a range of ligand valency demonstrates this interplay (Fig. {@fig:valency}b-d). Here, the ligand binding ratios for $R_1$ between cell populations are shown for ligands of high, medium, and low affinities ($K_d$ of $1000$, $100$, and $10$ $\mathrm{nM}$). The ligand binding ratio between $R_1^{hi}R_2^{lo}$ and $R_1^{med}R_2^{lo}$ is maximized by low affinity ligands, but requires greater valency to achieve peak binding selectivity when compared to ligands of greater affinity (Fig. {@fig:valency}b). A similar valency optimum for a given affinity is seen for binding selectivity between $R_1^{hi}R_2^{hi}$ and $R_1^{med}R_2^{med}$, and $R_1^{hi}R_2^{med}$ and $R_1^{med}R_2^{med}$ (Fig. {@fig:valency}c-d). Ligands with lower affinities achieve optimal binding with higher valencies and exhibit higher selectivity for cells expressing greater amounts of receptor.

Our model allowed us to identify the mechanism of valency-mediated receptor abundance selectivity by looking at the number of complex bound in each degree. A cell expressing $10^4$ receptors displays similar amounts of binding at each valency for ligands with dissociation constants of $1000$, $100$, and $10 \mathrm{nM}$ (Fig. {@fig:valency}e). However, a cell expressing 10% as many receptors exhibits extremely low amounts of multivalent binding for ligand complexes of low binding affinity (Fig. {@fig:valency}f). This effect arises due to a higher dissociation rate than multi-valent binding association rate at low receptor abundances (Fig. {@fig:valency}g). Multivalent ligands will undergo initial binding events at rates unaffected by receptor abundance, but low affinity and receptor density severely limit the possibility of secondary binding events. In contrast, cells with higher receptor abundance accumulate ligand binding by binding ligand at high valency, as the rate of multi-valent binding is greater than that of disassociation (Fig. {@fig:valency}f). This effect allows for valency to provide selectivity between cells having differences in receptor expression density, where affinity alone fails.

![**Valency provides selectivity based on receptor expression levels.** a) Heat/contour maps of multivalent ligands bound to cell populations given their expression profiles of two receptors. Multivalent ligand subunits bind to only $R_1$ with an dissociation constant of $330 \mathrm{nM}$, and do not bind to $R_2$. Complexes vary in valency from 1 to 16. Ligand concentration $L_0=1 \mathrm{nM}$; crosslinking constant $K_x^*=10^{-10}$. b-d) Ligand binding ratio between various cell populations for ligands of valency ranging from 1 to 16. b) Ligand bound ratio of $R_1^{hi}R_2^{lo}$ to $R_1^{med}R_2^{lo}$, c) $R_1^{hi}R_2^{hi}$ to $R_1^{med}R_2^{med}$, and d) $R_1^{hi}R_2^{med}$ to $R_1^{med}R_2^{med}$. e-f) Number of ligands bound at each degree of binding for cells exposed to octavalent ligand complexes composed of subunits with dissociation constants of $1000$, $100$, or $10 \mathrm{nM}$ for receptor 1. e) Number of octavalent complex bound at each degree for a cell with $10^4$ receptors. f) Number of octavalent complex bound at each degree for a cell with $10^3$ receptors. g) Fraction of forward to reverse binding rates for a multivalent ligand binding to a cell with varying amounts of receptor expression, following an initial binding event.](figure3.svg){#fig:valency}

### Non-Overlapping Ligand Targeting Drives a Limited Selectivity Enhancement of Mixtures

While most therapies rely on the action of a single molecular species, mixtures may enhance selectivity through combinations of actions [@doi:10.1038/nbt.1549]. Furthermore, some biologics inevitably act as mixtures of species through heterogeneity in glycosylation and the presence of endogenous ligands [@DOI:10.1002/rcm.3330]. Therefore, it is important to understand how mixtures of complexes influence the overall response.

To evaluate the contribution of mixtures, we evaluated model-predicted binding while varying the composition between ligands with either $R_1$ or $R_2$-preferred binding, respectively (Fig. {@fig:mixture}a). The trend that arises is very similar to an additive combination of the single ligand cases. This pattern highlights a key limitation of using mixtures for selectivity: selectivity between two populations varies monotonic with the composition, so any mixture combination is no better than using the more specific ligand entirely(Fig. {@fig:mixture}b).

While mixture engineering fails to enhance binding selectivity between two cell populations, it is potentially beneficial when considering two or more off-target cell populations. More specifically, when a target population expresses two target receptors, but off-target populations express each receptor individually in high amounts, drug mixtures can offer enhanced selectivity. For example, when maximizing targeting to $R_1^{hi}R_2^{hi}$ over $R_1^{hi}R_2^{lo}$ and $R_1^{lo}R_2^{hi}$, which individually express high levels of the receptors found on the $R_1^{hi}R_2^{hi}$, we show that a uniform mixture of ligands with high affinity for receptor 1 and 2 provides a modest improvement in targeting selectivity (Fig. {@fig:mixture}c). We also demonstrate the utility of mixture engineering in cell population combinations where many off-target populations are considered (Fig. {@fig:mixture}d). However, even in these cases, the magnitude of selectivity enhancement is modest. Finally, although we only consider the amount of binding, ligands can have non-overlapping signaling effects even with identical amounts of binding. In these cases, the effect of combinations can be distinct from either individual ligand [@pmid:28886385; @pmid:29960887].

![**Ligand mixtures with non-overlapping responses can enhance selectivity.** a) Heat/contour maps of multivalent ligands bound to cell populations given their expression profiles of two receptors. A mixture of monovalent ligands is used, with ligand 1 binding to receptor 1 and 2 with dissociation constants of $1 \mathrm{\mu M}$ and $10 \mathrm{\mu M}$ respectively, and ligand 2 binding to receptors 1 and 2 with dissociation constants of $10 \mathrm{\mu M}$ and $1 \mathrm{\mu M}$ respectively. Ligand concentration $L_0=1 \mathrm{nM}$; crosslinking constant $K_x^*=10^{-10}$. b-e) Ratio of ligand bound to cell populations exposed to monovalent mixtures of ligand one and 2. For b-d, the ratio of the target population to the single off target population with the greatest ligand bound is plotted. b) Ligand bound ratio of $R_1^{hi}R_2^{lo}$ to $R_1^{med}R_2^{lo}$, c) $R_1^{hi}R_2^{hi}$ to $R_1^{hi}R_2^{lo}$ and $R_1^{lo}R_2^{hi}$, d) $R_1^{med}R_2^{med}$ to $R_1^{med}R_2^{hi}$ and $R_1^{hi}R_2^{med}$.](figure4.svg){#fig:mixture}

### Heterovalent Bispecific Ligands Exhibit Unique Charateristics When Activated Fully Bound

Constructing multispecific drugs has become a promising new strategy for finer target cell specificity with the advancement of engineering techniques [@doi:10.1038/s41586-020-2168-1]. However, the number of possible configurations of multispecific drugs is combinatorially large and impossible to enumerate. Here, we use the heterovalent bispecific ligand as an example to explore the unique benefit of multispecificity distinct from any strategy analyzed before. We compared bispecificity with two aforementioned strategies: a 50%-50% mixture of two monovalent ligands, and a 50%-50% mixture of two different homogeneous bivalent ligands. These two strategies both have some similarities to bispecific therapeutics. A bispecific ligand contains two different ligand monomers with equal proportion, similar to a 50%-50% monovalent mixture. By comparing bispecific with a 50%-50% monovalent mixture, we can elucidate the extra benefit of tethering these two monomers into one complex. A bispecific ligand is also by nature bivalent, and by comparing it with homogeneous bivalent drugs, we can understand how having two different subunits in the same complex modify the behavior of a drug. Together, we sought to identify the unique advantage of heterovalency.

We first applied the binding model to predict the amount of ligand bound in bispecific drugs (Fig @fig:bispecific a), a 50%-50% mixture of two monomers (Fig @fig:bispecific b), and a 50%-50% mixture of two different homogeneous bivalents (Fig @fig:bispecific c), with the same set of parameters in ligand concentration and affinities. It turned out that the patterns of ligand bounds of these three cases are almost exactly the same, and bispecific doesn’t exhibit any merit. We then investigated if bispecific ligands that have to be activated with both of their subunits binding to a target display any special cell population selectivity characteristics. Bispecific ligands that require both subunits bound to be effective are common in the design of bispecific antibodies. To be specific, when the first antigen-binding region of a bispecific antibody docks at an antigen, it will only make a loose connection between the target and the antibody, unable to induce a strong immune response, and only when both antigen-binding regions bind to their target will this bispecific antibody be fully effective [@doi:10.1080/19420862.2015.1062192; @pmid:30145356].

To scrutinize the case where the target binding of both subunits is required, we extended our model to calculate the amount of ligand fully bound. For the heterovalent bispecific case, ligand fully bound will only account the amount of ligand with both of their ligand monomers bind to a target. With the same set of parameters, the predictions made for bispecific fully bound show a very distinct pattern from general ligand bound (Fig @fig:bispecific e). Besides that ligand fully bound has lower amounts, the contour plot of fully bound bispecific ligands has more concaved contour lines. For example, $R_1^{hi}R_2^{hi}$ has about the same level of general ligand bound as $R_1^{hi}R_2^{med}$ (Fig @fig:bispecific a), but it has significantly more ligand fully bound than $R_1^{hi}R_2^{med}$ (Fig @fig:bispecific e). This concavity of contour lines indicates that for bispecific complexes, ligand fully bound prefers double-positively expressed cells than singly-positively expressed cells. This should be obvious since these two subunits of the bispecific complex prefer different receptors. The results indicate that bispecific ligands will only exhibit special characteristics when it can only be activated fully bound. This characteristic opens up brand new engineering opportunities for targeting specific cell populations.

The specific amount of fully bound ligands depends on the propensity of crosslinking, captured by $K_x^*$. We plotted the pattern of bispecific fully bound with $K_x^* = 10^{-10}$, $10^{-12}$, and $10^{-14}$ (Fig. @fig:bispecific d-f). In general, when $K_x^*$ is larger, the ligands are more capable of multimerization, and there will be more fully bound units. To demonstrate how this characteristic of fully bound bispecific ligand imparts cell population specificity, we plotted the ratios of selectivities between some cell population pairs under bispecific versus another drugs given a range of $K_x^*$ (Figure @fig:bispecific g,h). These plotted numbers are ratios of ligand (fully) bound ratios between two cell populations, so when they are larger, it implies that fully bound bispecific ligand has more advantages than its counterpart to impart selective binding between these two cell populations. Figure @fig:bispecific g compares the selectivities under bispecific ligands versus a 50%-50% mixture of monovalent ligands. The results show that fully bound bispecific can grant better selective binding when $K_x^*$ is small enough. This logical, since when $K_x^*$ is small and crosslinking is rarer, most ligands will bind monovalently, and fully bound bispecific will especially favor the cell populations with higher receptor expression. However, when we compare fully bound bispecific to fully bound homogeneous bivalent mixtures (Fig. @fig:bispecific h), the advantage of bispecific drugs is not obvious except for $R_1^{med}R_2^{hi}$ to $R_1^{hi}R_2^{med}$ selectivity. Given that both therapeutics in comparison only account for fully bound ligands, the effect demonstrated in Figure @fig:bispecific g no longer holds. Together, we showed that bispecific ligands only exhibit unique advantages in inducing selective binding when they can only be effective when both of their subunits bind and crosslinking is more difficult.

![**Bispecific ligands exhibit unique effects when they can only be activated with both subunits bind.** a-c) Ordinary ligand bound are dominated by the initial binding of ligands, so it doesn't provide unique advantages to a) bispecific ligand comparing with b) a 50%-50% mixture of monovalent ligands and c) a 50%-50% mixture of bivalent ligands. Ligand concentration $L_0=10 \mathrm{nM}$; binding affinities $K_{d11}=100 \mathrm{nM}, K_{d22}=1 \mathrm{\mu M}, K_{d12}=K_{d21}=10 \mathrm{\mu M}$. d-f) The amount of fully bound bispecific ligands depends on the tendency of multimerization, capsuled by $K_x^*$. d) $K_x^* =10^{-10}$, e) $K_x^* =10^{-12}$, f) $K_x^* =10^{-14}$. g,h) The ratios of selectivities, thus "the ratios of binding ratios", varies with $K_x^*$. g) bispecific versus monovalent 50%-50% mixture, h) bispecific versus a 50%-50% homo-bivalent mixture.](figure5.svg){#fig:bispecific}

### Combining Strategies For Superior Selectivity

Each strategy above provided selectivity benefits in distinct situations, suggesting that they might synergistically improve selectivity when combined. We explored this potential synergy using mathematical optimization to determine the optimally cell type selective ligand for our theoretical cell populations. Here, we sought to determine the optimal ligand for particular cell populations while considering all other theoretical populations as off-target cells. We set our initial binding selectivity performance metric and optimize it within set bounds (Fig. @fig:combination). To simulate the effects that affinity engineering, we allowed ligand-receptor affinities to vary. To understand the effects of valency and mixture engineering on selectivity, we allowed ligand valency and mixtures to vary, while also allowing affinity to vary due to the previously noted reliance that these strategies have on receptor-ligand kinetics. Finally, all ligand parameters were allowed to vary to simulate the effects of combining therapeutic engineering approaches.

Optimizing a ligand for selectivity to $R_1^{lo}R_2^{hi}$ highlights a situation in which combined affinity and valency individually impart greater specificity, and optimal selectivity is achieved by combining these two strategies (Fig. {@fig:combination}a-f). Here selectivity is optimized by ligands with selective binding to receptor 2, and those with valencies allowing for selective binding to cells with higher abundances of receptor expression. One case contradictory to this trend is shown during the optimization for selectivity towards the $R_1^{hi}R_2^{hi}$ . While affinity, valency, and mixture engineering are able to impart some small contributions to enhanced selectivity, significant improvement is only achieved when combining them all (Fig. {@fig:combination}g-l). Here, synergistic combination of strategies can be employed to great effect, granting far superior ligand selectivity. A more difficult design problem is featured in the optimization of $R_1^{med}R_2^{med}$. Since it lies in the midst of the other populations in receptor expression space (Fig. {@fig:mixture}m-r), any modulation of affinity, valency or combining it with mixture-based strategies seems ineffective.

Our results highlight that both in singular and combined strategies for therapeutic manipulation, the target and off-target populations dictate the optimal approach. It is also clear that combined approaches do offer synergies which can be harnessed, but that those are only emergent in particular therapeutic situations [@pmid:30145356].

![**Combinations of strategies provide superior selectivity.** a,g,m) Optimal selectivity levels (average ligand bound compared to average ligand bound by all other populations) achieved using various ligand engineering techniques. Ligand concentration $L_0=1 \mathrm{nM}$. Xnot ligands ligands are monovalent ligands with affinities of $1 \mathrm{\mu M}$ for both receptor 1 and 2. The dissociation constant was allowed to vary between $10 \mathrm{mM}$ and $0.1 \mathrm{nM}$ for both receptors using the "affinity" approach. Valency was allowed to vary from 1 to 16 for the "valency" approach in addition to affinities varying. Mixtures were assumed to be composed of two monovalent ligands, and affinities were allowed to vary in the "mixture" approach. The combined "all" approach allowed all of these quantities to vary simultaneously. The crosslinking constant $K_x^*$ was allowed to vary between $10^{-15}$ and $10^{-9}$ for all approaches. b-f,h-l,n-f) Heatmap of magnitude of ligand bound for ligand with optimized characteristics according to various ligand engineering strategies. Target population is shown in red. a-f) Pertains to optimal targeting of $R_1^{lo}R_2^{hi}$, g-l) pertains to optimal targeting of $R_1^{hi}R_2^{hi}$, and m-r) pertains to optimal targeting of $R_1^{med}R_2^{med}$.](figure6.svg){#fig:combination}

### Using Binding Competition to Invert Receptor Targeting

While the strategies above provided selectivity in many cases, we recognized that they are all limited to a positive relationship between receptor abundance and binding. Therefore, we wondered if binding competition with a receptor antagonist, or "dead ligand", could invert this relationship.

To investigate the effect of ligand competition with an antagonist, we modeled mixtures of ligands, but only quantified the amount of binding for the active ligand. As binding is dependent upon the properties of both ligands, we chose to only consider a monovalent agonist and tetravalent antagonist (Fig. {@fig:deadLig}). We found that combinations of monovalent agonistic ligands and multivalent antagonistic ligands were able to uniquely target populations expressing small or intermediate amounts of receptors, which is demonstrated when comparing ligand binding ratios between $R_1^{med}R_2^{lo}$ to $R_1^{hi}R_2^{lo}$ (Fig. {@fig:deadLig}a). Here, a nearly tenfold increase in selectivity can be granted to monovalent agonists when combined with a tetravalent antagonist. In this case, there are greater quantities of agonist bound to $R_1^{med}R_2^{lo}$ than $R_1^{hi}R_2^{lo}$ (Fig. {@fig:deadLig}b). This is striking, as $R_1^{med}R_2^{lo}$ expresses either as many or fewer abundances of receptors one and two when compared to $R_1^{hi}R_2^{lo}$. This phenomenon, which could not be achieved without multivalent antagonists, occurs due to the preferential binding of multivalent antagonists to populations expressing higher abundances of receptors (Fig. {@fig:deadLig}c). Thus, in cases where previously discussed ligand engineering strategies and approaches fail to achieve selective binding to cells expressing smaller or similar amounts of receptors to off-target populations, combinations of agonistic and antagonistic ligands may provide unique benefits.

![**Mixtures of receptor agonists and antagonists allow for unique population targeting activity.** Ligand concentration $L_0=1 \mathrm{nM}$. a) Selectivity for $R_1^{med}R_2^{lo}$ against $R_1^{hi}R_2^{lo}$ when exposed to a tetravalent "dead ligand" antagonist with varying affinities for receptors 1 and 2, and a monovalent therapeutic receptor agonists with affinities optimized for selectivity. Heatmap values are normalized to specificity imparted with non-binding antagonists. Only amount of agonist bound is considered in determination of selectivity. b-d) Heatmap of agonist (b), and antagonist (c) ligand bound for antagonist and agonist ligand combination shown to imaprt greatest selectivity improvement in (a). d) Heatmap of agonist bound in b,c when no antagonist is present.](figure7.svg){#fig:deadLig}

## Discussion

<!-- Summary. -->

Here, we developed and employed a multivalent, multi-ligand, multi-receptor binding model to explore the effectiveness of various ligand engineering strategies for population-selective binding (Fig. {@fig:model}). We compared a few different strategies, including changes to affinity, ligand valency, mixtures of species, multi-specificity, and these in combination. Our analysis was performed using a representative set of theoretical cell populations defined by their distinct expression of two receptors. We identified that each strategy helped to enhance specificity, but only in certain situations. The specific benefits of each strategy meant that their combination led to synergistic specificity enhancements.

Each strategy’s contribution could be summarized by general patterns. We found that affinity changes were most effective when the target and off-target populations expressed distinct combinations receptors (Fig. {@fig:affinity}). Increasing the affinity of the receptors that are more abundantly expressed on the target cells, and decreasing those that are not, increased the binding selectivity, with greater differences in receptor patterns corresponding to greater benefits. When target and off-target populations had similar receptor expression profiles and only differed in amounts, valency, but not affinity, changes were effective (Fig. {@fig:valency}). The efficacy of valency changes was greatly dependent on the kinetics of the receptor-ligand interactions and the difference in receptor abundance as the secondary binding and unbinding rates were a key determinant. Mixtures of therapeutic ligands were found to be mostly ineffective in imparting binding selectivity, but had modest benefits with multiple off-target populations (Fig. {@fig:mixture}). Heterovalent bispecific ligands only showed unique advantages over mixtures of monovalent ligands or bivalent ligands when we restricted our binding quantity to be both subunits bound (Fig. {@fig:bispecific}). They prefer target populations that have high expression of both receptors over those with high expression of only a single receptor, with the preference for secondary binding as the key determinant for the magnitude of advancement. We found that, while a single ligand engineering strategy dominated in its contributions to cell type selectivity, synergies between these strategies existed in some cases to derive even greater specificity (Fig. {@fig:combination}). Finally, we found that combinations of monovalent therapeutic ligands with multivalent antagonistic ligands allowed improved selectivity for target populations expressing relatively fewer receptors to off-target populations.

<!-- TODO: Mention more than two receptors, but should come below. -->

<!-- A number of strategies are already employed. -->

A number of strategies are already utilized in existing engineered therapies. For example, affinity changes in the cytokine IL-2 have been used to bias its effects towards either effector or regulatory immune populations [@doi:10.1158/2159-8290.CD-18-1495; @pmid:30446251]. Varying the valency of tumor-targeted antibodies leads to targeted cell clearance based upon the levels of expressed antigen [@doi:10.1038/srep40098]. Manipulation of the affinities of fibronectin domains displayed by octovalent nanorings was shown to enhance the selectivity of binding to cancerous cells displaying relatively higher densities of fibronectin receptors compared to native tissue [@doi:10.1021/jacs.8b09198]. The advantages offered by therapeutics reliant on low-affinity, multivalent interactions to selectively bind cells with high receptor expression densities were also described in a study considering a therapeutic reliant on multivalent complement binding to small-molecule therapeutics with αvβ3 binding domains, which is expressed at higher densities on tumor cells, and ⍺-Gal binding domains which are recognized by multivalent human anti-α-galactosyl antibodies [@doi:10.1021/cb6003788]. These examples lend support to the accuracy of our model. At the same time, recognizing the previously described ligand engineering approaches as separable strategies provides clearer guidance for future engineering.

<!-- Still need to implement others. -->

Nonetheless, many optimization strategies described in this study have not been exploited in pharmaceutical practice yet, not because of the lack of consideration but due to the complexity of real biological contexts. First, some strategies are not practical biochemically. For example, the manipulation of ligand affinity requires intricate protein engineering. While the direction of affinity changes can be predicted, dictating the affinity of an engineered molecular a priori is almost unattainable. Moreover, it is very likely altering the affinity of a ligand to the receptor of interest can change the affinity of it to other receptors, which may offset the intended benefit in improving selectivity. Also, the change in the receptor expression profile of a cell population complicates the matter. It is well documented that cancer cells can escape therapeutic targeting by upregulating [@doi:10.1016/j.devcel.2019.07.010; @doi:10.1158/0008-5472.CAN-13-0602] or downregulating [@pmid:14534734] certain receptors. In this case, not only should the abundance of receptors be considered, but also their variance within a cell population over time. The effects of expression change on the efficacy of a therapeutic are not evident without a mechanism-based analysis. Furthermore, while we explore the efficacy of combining monovalent agonists with multivalent antagonists for targeting cells expressing relatively fewer receptors to off-target populations, the complexity of engineering the affinities of both ligands makes effective implementation of the strategy much more difficult in practice. While this work does not intend to solve every aspect of these issues, we propose that using a computational binding model can analyze these strategies quantitatively and collectively from a mechanistic perspective. Even when the absolute mathematical optimum cannot be achieved biochemically, our analyses provide good guidance to what is attainable and how to approach the optimum, accounting those biological complexities and facilitating the implementation of complicated combinations of several strategies.

<!-- No strategies for NOT relationships. -->

While the exploration of our multivalent binding model has elucidated strategies for selective targeting of many populations with respect to one or many off-target populations, several inter-population receptor expression relationships remain challenging to impart selectivity to. For example, target populations that unilaterally express fewer receptors of every type than off-target populations present a unique challenge for the design of a selective ligand. While we computationally show the potential for the simultaneous use of multivalent "dead" ligands, which bind but do not induce a response, and monovalent "live" ligands to exploit reduced receptor expression levels, the translation of such a therapeutic to a clinical application remains questionable (Fig. {@fig:combination} s-v). In cases where a target population expresses fewer receptors of any kind than an off-target population, our analysis suggests that other target receptors or therapeutic strategies should be considered. However, in cases where target populations express more of any type of receptor than an off-target population, we show that one or more of our formulated ligand engineering strategies can be employed to improve binding selectivity.

<!-- Impressive range of logic can be built without cells involved. -->

In many therapeutic applications where selective engagement of target cell populations is an important performance metric, such as the treatment of cancer, cellular engineering is becoming increasingly popular [@doi:10.1146/annurev-immunol-042718-041407]. Human engineered chimeric antigen receptor (CAR) T cells are being introduced in the clinical setting at a rapid rate, and advances in genetic engineering technologies have enhanced the potential of engineered immune cells to recognize and attack malignant tissues [@doi:10.1016/j.ymthe.2017.06.012]. These technologies bypass ligand-receptor binding considerations by manipulating the complexities of the immune system to attack cells expressing particular receptors. However, manipulations of the immune system inevitably lead to myriad unintended effects due to the complex regulatory mechanisms which control immune activation and activity. We have shown here that complex and targeted changes to the binding activity of simple therapeutic ligands can be achieved via simple ligand characteristic manipulation. These manipulations can be generally applied to achieve greater selectivity, as well as more precisely guided using a computational binding model. The impressive array of logic described in this study can be applied to guide rational ligand engineering applications. It should be noted that the application of this logic is reliant on knowledge of the target and off-target cell population receptor expression levels. Future application of the ligand binding logic described in this study could be guided using high throughputs single-cell profiling techniques, such as RNA-seq or high-parameter flow cytometry. A computational tool that could translate such data-sets into immediately applicable ligand design criteria may represent a potential avenue for the translation of our analyses into a more broadly applicable ligand engineering tool.


## Materials and Methods

### Data and Software Availability

All analysis was implemented in Python v3.8, and can be found at <https://github.com/meyer-lab/cell-selective-ligands>.

### Generalized multi-ligand, multi-receptor multivalent binding model

To model multivalent ligand complexes, we extended our previous binding model to the multi-ligand case [@pmid:29960887]. We define $N_L$ as the number of distinct monomer ligands, $N_R$ the number of distinct receptors, and the association constant of monovalent binding between ligand $i$ and receptor $j$ as $K_{a,ij}$. Multivalent binding interactions after the initial interaction have an association constant of $K_x^* K_{a,ij}$, proportional to their corresponding monovalent affinity. The concentration of complexes is $L_0$, and the complexes consist of random ligand monomer assortments accoring to their relative proportion. The proportion of ligand $i$ in all monomers is $C_i$. By this setup, we know $\sum_{i=1}^{N_L} C_i = 1$. $R_{\mathrm{tot},i}$ is the total number of receptor $i$ expressed on the cell surface, and $R_{\mathrm{eq},i}$ the number of unbound receptors $i$ on a cell at the equilibrium state during the ligand complex-receptor interaction.

The binding configuration at the equilibrium state between an individual complex and a cell expressing various receptors can be described as a vector $\mathbf{q} = (q_{1,0}, q_{1,1}, ..., q_{1,N_R}, q_{2,0},..., q_{2,N_R},q_{3,0},..., q_{N_L, N_R})$ of length $N_L(N_R+1)$, where $q_{i,j}$ is the number of ligand $i$ bound to receptor $j$, and $q_{i,0}$ is the number of unbound ligand $i$ on that complex in this configuration. The sum of elements in $\mathbf{q}$ is equal to $f$ , the effective avidity. For all $i$ in $\{1,2,..., N_L\}$, let $φ_{i,j} = R_{\mathrm{eq},j} K_{a,ij} K_x^* C_i$ when $j$ is in $\{1,2,...,N_R\}$, and $φ_{i,0} = C_i$. The relative amount of complexes in the configuration described by $\mathbf{q}$ at equilibrium is


$$v_{\mathbf{q},eq} = \binom{f}{\mathbf{q}} \frac{L_0}{K_x^* } \prod_{i=1\\ j=0}^{i=N_L\\ j=N_R}{{φ_{ij}}^{q_{ij}}},$$


with $\binom{f}{\mathbf{q}}$ being the multinomial coefficient. Then the total relative amount of bound receptor type $n$ at equilibrium is


$$ R_{\mathrm{bound},n} = \frac{L_0 f}{K_x^* } \sum_{m=0}^{N_L}φ_{mn} \left(\sum_{i=1\\ j=0}^{i=N_L\\ j=N_R}{{φ_{ij}}^{q_{ij}}}\right)^{f-1} .$$


By conservation of mass, we know that $R_{\mathrm{tot},n} = R_{\mathrm{eq},n} + R_{\mathrm{bound},n}$ for each receptor type $n$, while $R_{\mathrm{bound},n}$ is a function of $R_{\mathrm{eq},n}$. Therefore, each $R_{\mathrm{eq},n}$ can be solved numerically using $R_{\mathrm{tot},n}$. Similarly, the total relative amount of complexes bind to at least one receptor on the cell is


$$ L_{\mathrm{bound}} = \frac{L_0}{K_x^* } \left [\left(\sum_{i=1\\ j=0}^{i=N_L\\ j=N_R}{{φ_{ij}}^{q_{ij}}}\right)^f -1 \right] .$$


### Generalized multivalent binding model with defined complexes

When complexes are engineered and so ligands are not randomly sorted into multivalent complexes, such as with the Fabs of bispecific antibodies, the proportions of each kind of complexes become exogenous variables and are no longer decided by the monomer composition $C_i$'s. The monomer composition of a ligand complex can be represented by a vector $\mathbf{θ} = (θ_1, θ_2, ..., θ_{N_L})$, where each $θ_i$ is the number of monomer ligand type $i$ on that complex. Let $C_{\mathbf{θ}}$ be the proportion of the $\mathbf{θ}$ complexes in all ligand complexes, and $Θ$ be the set of all possible $\mathbf{θ}$'s. We have $\sum_{\mathbf{θ} \in Θ} C_{\mathbf{θ}} = 1$.

The binding between a ligand complex and a cell expressing several types of receptors can still be represented by a series of $q_{ij}$. The relationship between $q_{ij}$'s and $θ_i$ is given by $θ_i = q_{i0} + q_{i1} + ... + q_{iN_R}$. Let the vector $\mathbf{q}_i = (q_{i0}, q_{i1}, ..., q_{iN_R})$, and the corresponding $\mathbf{θ}$ of a binding configuration $\mathbf{q}$ be $\mathbf{θ}(\mathbf{q})$. For all $i$ in $\{1,2,...,N_L\}$, we define $ψ_{ij} = R_{\mathrm{eq},j} K_{a,ij} K_x^*$ where $j = \{1,2,...,N_R\}$ and $ψ_{i0} = 1$. The relative amount of complexes bound to a cell with configuration $\mathbf{q}$ at equilibrium is


$$v_{\mathbf{q},eq} = \frac{L_0 C_{\mathbf{θ}(\mathbf{q})}}{K_x^* }
\prod_{i=1\\j=0}^{i=N_L\\ j=N_R} {ψ_{ij}}^{q_{ij}}
\prod_{i=1}^{N_L} \binom{θ_i}{\mathbf{q}_i} .$$


Then we can calculate the relative amount of bound receptor $n$ as


$$
R_{\mathrm{bound},n} = \frac{L_0}{K_x^* } \sum_{\mathbf{θ} \in Θ} C_{\mathbf{θ}}
\left[ \sum_{i=1}^{N_L} \frac{ψ_{in} θ_i}{\sum_{j=0}^{N_R} ψ_{ij}} \right]
\prod_{i=1}^{N_L} \left( \sum_{j=0}^{N_R} ψ_{ij}\right)^{θ_i} .
$$


By $R_{\mathrm{tot},n} = R_{\mathrm{eq},n} + R_{\mathrm{bound},n}$, we can solve $R_{\mathrm{eq},n}$ numerically for each type of receptor. The total relative amount of ligand binding at equilibrium is


$$ L_{\mathrm{bound}} =  \frac{L_0}{K_x^* } \sum_{\mathbf{θ} \in Θ} C_{\mathbf{θ}}
\left[ \prod_{i=1}^{N_L} \left( \sum_{j=0}^{N_R} ψ_{ij}\right)^{θ_i} -1 \right] .$$


### Mathematical optimization

We used the SciPy function `scipy.optimize.minimize` to combine several strategies and achieve optimal selectivity [@doi:10.1038/s41592-019-0686-2]. Unless specified otherwise, the initial values for optimization were $10^{-12}$ for crosslinking coefficient $K_x^*$, 1 for valency $f$, 100% for mixture composition, and $10^6 \mathrm{M^{-1}}$ for the affinity constants; the boundaries were $10^{-15} -- 10^{-9}$ for $K_x^*$, 1–16 for $f$, 0–100% for mixture composition, and $10^2 -- 10^{10} \mathrm{M^{-1}}$ for the affinity constants.

### Sigma point filter

To consider the intrapopulation variance of a cell population in the optimization, we implemented sigma point filter [@wikidata:Q99353631], a computationally efficient method to approximate the variance proprogated through an ordinary differential equation-based model.

### Reimplementation of Csizmar et al.

To validate our model, we recapitulated multivalent binding data from Csizmar et al. using our multivalent binding model (Fig. S1) [@doi:10.1021/jacs.8b09198]. Here, fluorescently labeled nanorings displaying 1, 2, 4, and 8 fibronectin clones, which bind to epithelial cell adhesion molecule (EpCAM) antigens, were fabricated. Binding activity of nanorings displaying both high (C5) and low (B22) affinity EpCAM binding domains was measured. Binding to an EpCAM^high^ ($3.3 \times 10^6$ antigens/cell) population was measured using flow cytometry. We used nonlinear least squares optimization, as described above, to fit our multivalent binding model to the binding data, using a unique scaling factor for each fibronectin clone to convert between measured fluorescent intensity and magnitude of ligand bound. We allowed affinity of the fibronectin clones to vary during optimization.


## Supplementary Figures

![**Model is able to recapitulate experimental multivalent binding activity.** a) Experimental vs. predicted fluorescent units for MCF-7 cells expressing $3.8 \times 10^{6}$ fibronectin receptors per cell bound to nanorings expressing one, two, four or eight fibronectin binding domains at concentrations of 0.16–500 nM. C5 and B22 are high and low affinity fibronectin binding domains respectively. Using nonlinear least-squares regression, the crosslinking coeffient was found to be $1.11 \times 10^{-12}$; the association constants for C5 and B22 were found to be $5.78 \times 10^{-5}$, $3.08 \times 10^{-12}$ respectively, and ligand to fluorescent conversion factor for C5 and B22 were found to be $5.0 \times 10^{-2}$ and $3.2 \times 10^{-2}$ respectively. b) Predicted fluorescent values for MDA cells expressing $5.2 \times 10^{4}$, SK cells expressing $2.2 \times 10^{5}$, LNCaP cells expressing $2.8 \times 10^{6}$, and MCF-7 cells expressing $3.8 \times 10^{6}$ receptors per cell as described in Csizmar et al. [@doi:10.1021/jacs.8b09198]. Fluorescence was predicted for interactions with octa- and tetravalent ligands expressing either C5 or B22 fibronecting binding domains. c) Predicted binding ratios of MCF-7 cells to MDA cells when bound with octa- and tetravalent ligands expressing either C5 or B22 fibronecting binding domains.](figureS1.svg){#fig:Csizmar}


## Acknowledgements

This work was supported by NIH U01-AI148119 to A.S.M. **Competing financial interests:** The authors declare no competing financial interests.



## Author contributions statement

A.S.M. concieved of the work. All authors implemented the analysis and wrote the paper.


## References {.page_break_before}

<!-- Explicitly insert bibliography here -->
<div id="refs"></div>
