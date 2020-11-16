---
author-meta:
- Brian Orcutt-Jahns
- Zhixin Cyrillus Tan
- Aaron S. Meyer
bibliography: []
date-meta: '2020-11-16'
header-includes: '<!--

  Manubot generated metadata rendered from header-includes-template.html.

  Suggest improvements at https://github.com/manubot/manubot/blob/master/manubot/process/header-includes-template.html

  -->

  <meta name="dc.format" content="text/html" />

  <meta name="dc.title" content="A quantitative view of strategies to engineer cell-selective ligand binding" />

  <meta name="citation_title" content="A quantitative view of strategies to engineer cell-selective ligand binding" />

  <meta property="og:title" content="A quantitative view of strategies to engineer cell-selective ligand binding" />

  <meta property="twitter:title" content="A quantitative view of strategies to engineer cell-selective ligand binding" />

  <meta name="dc.date" content="2020-11-16" />

  <meta name="citation_publication_date" content="2020-11-16" />

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

  <link rel="alternate" type="text/html" href="https://meyer-lab.github.io/cell-selective-ligands/v/fa4b9665ee4fd073b4ae100b4d793370bee55af0/" />

  <meta name="manubot_html_url_versioned" content="https://meyer-lab.github.io/cell-selective-ligands/v/fa4b9665ee4fd073b4ae100b4d793370bee55af0/" />

  <meta name="manubot_pdf_url_versioned" content="https://meyer-lab.github.io/cell-selective-ligands/v/fa4b9665ee4fd073b4ae100b4d793370bee55af0/manuscript.pdf" />

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
was automatically generated on November 16, 2020.
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

A critical property of many therapies is their selective binding to specific target populations. Exceptional specificity can arise from high-affinity binding to unique cell surface targets. In many cases, however, therapeutic targets are only expressed at subtly different levels relative to off-target cells. More complex binding strategies have been developed to overcome this limitation, including multi-specific and multi-valent molecules, but these create a combinatorial explosion of design possibilities. Therefore, guiding strategies for developing cell-specific binding are critical to employ these tools. Here, we extend a multi-valent binding model to multi-ligand and multi-receptor interactions. Using this model, we explore a series of mechanisms to engineer cell selectivity, including mixtures of molecules, affinity adjustments, and valency changes. Each of these strategies maximizes selectivity in distinct cases, leading to synergistic improvements when used in combination. Finally, we identify situations in which selectivity cannot be derived through passive binding alone to highlight areas in need of new developments. In total, this work uses a quantitative model to unify a comprehensive set of design guidelines for engineering cell-specific therapies.

## Summary points

- Affinity, valency, and other alterations to target cell binding provide enhanced selectivity in specific situations.
- Evidence for the effectiveness and limitations of each strategy are abundant within the drug development literature.
- Combining strategies can offer enhanced selectivity.
- A simple, multivalent ligand-receptor binding model can help to direct therapeutic engineering.



## Introduction

<!-- Targeting specific cell populations is a universal challenge in protein therapies. -->

Many drugs both derive their therapeutic benefit and avoid toxicity through selective binding to specific cells within the body. Often, target cells can differ from off-target populations only subtly in surface receptor expression, making selective activation of target cells difficult to achieve. Even with a drug of very specific molecular binding, genetic and non-genetic heterogeneity can create a wide distribution of cell responses. This can result in reduced effectiveness and increased toxicity. Specific cell targeting is a universal challenge in protein-based therapies. For example, in cancer, resistance to anti-tumor antibodies [@doi:10.1158/1078-0432.CCR-09-1735], targeted inhibitors [@pmid:20129249], chemotherapies [@doi:10.1016/j.cell.2016.03.025], and chimeric antigen receptor T cells [@doi:10.1056/NEJMoa1407222; @doi:10.1126/scitranslmed.aaa0984] all can arise through the selection among heterogeneous cell populations. While the immune system takes advantage of heterogeneity at the single-cell level to translate noisy inflammatory signals into robust yet sensitive responses [@PMID:24919153], this heterogeneity impedes our effort of creating a highly specific drug. The intricacies of both inter-population receptor expression difference and intra-population receptor expression heterogeneity present significant challenges that limit the selectivity of therapies within the body.

<!-- Need alternative strategies for engineering specificity. -->

Further improving cell-specific targeting will require new strategies of engineering specificity. Non-cellular therapies such as protein therapies have most extensively been engineered to target specific cell types through mutations that provide high-affinity binding to unique surface antigens [@pmid:25992859]. This strategy can enhance specificity, but only to a limited degree, particularly when target cells can only be distinguished by subtle quantitative differences in surface antigen or by combinations of markers. The limitations of single-antigen targeting have led to efforts to engineer logic program and more complex programs into cellular therapies to recognize target cells more specifically [@doi:10.1126/science.aay2790; @pmid:30889382; @doi:10.1016/j.cell.2018.03.038]. However, non-cellular therapies have considerable benefits in drug access, manufacturing, and reliability; some of the same benefits have begun to be engineered into these agents.

<!-- We need computational models to make sense of this. -->
The enormous number of potential configurations of ligand designs make computational tools essential for designing highly selective therapies [@doi:10.1101/812131]. Here, we analyze a suite of molecular approaches for engineering cell-specific binding using a multi-valent, multi-receptor, multi-ligand model. We show that strategies including affinity, valency, binding competition, ligand mixtures, and hetero-valent complexes provide distinct improvements in cell-specific targeting. Finally, we combine these strategies to target cells through combinatorial methods. In total, our results demonstrate that binding programs can offer combinatorial targeting strategies with similar effectiveness as complex engineered cellular therapies and can be engineered using a mechanistic binding model.


## Results

### A Model System to Explore the Factors Contributing to Cell Selectivity

![**A model system for exploring the factors contributing to cell selectivity.** a) A simplified schematic of the binding model. There are two types of receptors and two types of ligand monomers that form a tetravalent complex. b) A cartoon for four cell populations expressing two different receptors at low or high amounts. c) A sample heat/contour map for the model-predicted log ligand bound given the expression of two types of receptors. d) Eight arbitrary theoretical cell populations with various receptor expression profiles.](figure1.svg){#fig:model}

Here we investigate cell-specific targeting quantitatively by extending a multi-valent, multi-ligand equilibrium binding model. Virtually any therapy, including monoclonal antibodies, small-molecule inhibitors, and cytokines, can be thought of as ligands for respective receptors expressed on target cells. Ligand binding to target cells is the first step and essential for a drug's intended action; in contrast, binding to off-target cells can result in unintended effects or toxicity. Some cell populations can be distinguished by their expression of a unique receptor not expressed by other populations, but more commonly, target and off-target cells express the same collection of receptors and differ only in their magnitudes of receptor expression. In such situations, engineering drugs to optimize target cell binding while minimizing their off-target cells binding is an area of ongoing research and has inspired a myriad of drug design strategies [@doi:10.1038/sj.bjc.6604700; @doi:10.1021/cb6003788; @doi:10.1038/s41586-018-0830-7]. In this work, we define cell population selectivity as the ratio of the number of ligands bound to target cell populations divided by the number of ligands bound to off-target cell populations. We will use a quantitative binding estimation for each cell population to examine these strategies.

While ligand-receptor binding events in biology are diverse, they are governed by thermodynamic properties and the law of mass action. For the binding between monomer ligands and receptors, their affinity can be described by the association constant $K_a$, or its reciprocal, the dissociation constant $K_d$. Binding behavior is more complicated when the ligands are multivalent complexes consisting of multiple units, each of which can bind to a receptor (Fig. {@fig:model}a). During initial association, we assume that the first subunit on a ligand complex binds according to the same dynamics that govern monovalent binding. Subsequent binding events exhibit different behavior, however, due to the increased local concentration of the complex and steric effects. Here, we assume that the effective association constant for the subsequent bindings is proportional to that of the free binding, but scaled by a crosslinking constant, $K_x^*$, which describes how easily a multivalent ligand bound to a cell monovalently can attain secondary binding. Another consideration that must be made when modeling multivalent binding processes is whether ligand complexes are formed via random assortment of monomers, or whether the monomer composition is uniform across complexes by engineering. We developed a multivalent binding model that calculates the amount of ligand bound at equilibrium taking each of these factors into account (see methods).

As a simplification, we will consider theoretical cell populations that express only two receptors capable of binding ligand (Fig. {@fig:model}b), ranging in abundance from 100 to 10,000 per cell. Figure {@fig:model}c shows the log-scaled predicted amount of binding of a monovalent ligand given the abundance of two receptors, with the concentration of ligand, $L_0$, to be $1 \mathrm{nM}$, and its dissociation constants to the two receptors to be $10 \mathrm{\mu M}$ and $100$ $\mathrm{nM}$, respectively. Because all axes are log-scaled, the number of contour lines between two points indicates the ratio of ligand binding between populations. For instance, in figure {@fig:model}c, cell populations at points 1 and 2 are on the same contour line and thus have the same amount of ligand bound; the cell populations at points 1 and 3 are separated by multiple contour lines, indicating that cells at point 3 bind more ligands (In fact, the ratio can be read as the exponent of the contour line difference. For point 3 to point 1, the ratio is $e ^{4.6-2.3} \approx 7.4$). Alternatively, we can think of moving from one point to another as a change of expression profile for a cell population. This situation might correspond to some cues inducing expression of a receptor, such as interferon-induced upregulation of MHC and regulatory T cell upregulation of IL-2R⍺ [@DOI:10.1073/pnas.0812851107]. When the amount of receptor 1 ($R_1$) increases on a cell (a point moves rightward, e.g. from 1 to 2), the amount of binding doesn’t increase significantly. On the contrary, a cell with increased expression of $R_2$ (moving upward, e.g. from 1 to 3) will bind significantly more ligands. Here, the ligand's high affinity for $R_2$ and relatively low affinity for $R_1$ lead to binding varying more strongly with changes to $R_2$ expression than $R_1$. 

To analyze more general cases, we arbitrarily defined eight theoretical cell populations according to their differential expression of two receptor types ($R_1$ and $R_2$ plotted on x and y axes). As shown in Fig {@fig:model}d, they either have high ($10^4$), medium ($10^3$), or low ($10^2$) expression of $R_1$ and $R_2$. The receptor expression profile within each cell population can also vary widely.  To demonstrate cell-to-cell heterogeneity, we also defined intrapopulation variability of each population arbitrarily. For instance, the expression profile of $R_1^{med} R_2^{med}$ has a wider range. The intrapopulation variance will be accounted for with sigma point filter. We will use this binding model to examine how engineering a ligand using various strategies can improve cell-specific targeting. Although we will only consider two receptor and ligand subunit types respectively, the principles we present can generalize to more complex cases.

### Affinity Provides Selectivity Toward Cell Populations with Divergent Receptor Expression

![**Affinity provides selectivity to cell populations with divergent receptor expression.** a) Heat/contour maps of monovalent ligand binding to cell populations given the surface abundance of two receptors. Ligand dissociation constants to these receptors range from $10 \mathrm{\mu M} \sim 100 \mathrm{nM}$. Ligand concentration $L_0=1 \mathrm{nM}$. b-e) Heatmap of binding ratio of cell populations exposed to a monovalent ligand with dissociation constants to receptor 1 and 2 ranging from $10^{4} \sim 10^{2} \mathrm{nM}$, at a concentration $L_0 = 1 \mathrm{nM}$. Ligand bound ratio of (b) $R_1^{hi}R_2^{lo}$ to $R_1^{lo}R_2^{hi}$, (c) $R_1^{med}R_2^{hi}$ to $R_1^{hi}R_2^{med}$, (d) $R_1^{hi}R_2^{lo}$ to $R_1^{med}R_2^{lo}$, and (e) $R_1^{hi}R_2^{hi}$ to $R_1^{med}R_2^{med}$.](figure2.svg){#fig:affinity}

We first explored altering the ligand binding affinity as an engineering strategy to enhance its cell population specificity. Here, we showed the binding pattern of monovalent ligands with various affinities ranging from $10 \mathrm{\mu M}$ to $100 \mathrm{nM}$ to both receptors $R_1$ and $R_2$ (Fig. {@fig:affinity}a). We found that when a target cell population expresses a receptor not expressed by off-target cell populations, enhancing the affinity of the drug to this receptor is a clear strategy to increase selective binding to this population. For example, $R_1^{hi}R_2^{lo}$ only significantly expresses $R_1$, while $R_1^{lo}R_2^{hi}$ only significantly expresses $R_2$. When the affinity to $R_1$ is enhanced and the affinity to $R_2$ is reduced, the binding selectivity towards $R_1^{hi}R_2^{lo}$ increases (Fig. {@fig:affinity}b). The contour plots in Fig. {@fig:affinity}a shows this trend more intuitively: when ligand affinity to $R_1$ increases, which corresponds to shifting from subplots on the left to the ones on the right, binding is shown to vary more strongly according to $R_1$ expression, indicating an increase in the amount of ligand bound for the populations with high $R_1$ expression, such as $R_1^{hi}R_2^{hi}$, $R_1^{hi}R_2^{med}$, and $R_1^{hi}R_2^{lo}$.

However, situations where both on- and off-target cell populations express the same set of receptors and differ only in their magnitude of expression are just as common. In these cases, we found out that it is beneficial for the drug to bind tightly to the most comparatively highly expressed receptor on the target population. Some examples of this pattern are the selective binding to $R_1^{med}R_2^{hi}$ over $R_1^{hi}R_2^{med}$, and $R_1^{hi}R_2^{lo}$ over $R_1^{med}R_2^{lo}$ (Fig. {@fig:affinity}a). For these two pairs, the benefit of affinity changes is limited by the relative discrepancy in receptor amounts (Fig. {@fig:affinity}c,d). Becuase the ratios of receptor expressions between $R_1^{med}R_2^{hi}$ and $R_1^{hi}R_2^{med}$ are significantly lower than those between $R_1^{hi}R_2^{lo}$ and $R_1^{lo}R_2^{hi}$, the greatest binding selectivity that can be achieved for these two populations is also significantly lower. When both receptors are uniformally more abundant in a target population than the off target population, such as when comparing $R_1^{hi}R_2^{hi}$ to $R_1^{med}R_2^{med}$, affinity tuning fails to modulate binding selectivity (Fig. {@fig:affinity}d). Therefore, to use affinity changes for selectivity enhancement, it is critical to identify which receptors a cell population of interest expresses the most uniquely when compared to off-target populations.

Our binding contour plots are also useful for considering the interplay between affinity modulation and intrapopulation receptor expression heterogeneity. For example, $R_1^{med}R_2^{med}$ has relatively high variance in both $R_1$ and $R_2$ expression (Fig. {@fig:affinity}a). When the binding affinities to $R_1$ and $R_2$ are divergent, as shown in the subplots in the upper left corner and bottom right corner of Fig. {@fig:affinity}a, the ellipse representing its range of receptor abundances rides through multiple contour lines, indicating wide variation in the quantity of bound ligand. This intrapopulation variation in the amount of ligand bound, however, is not observed when the affinities to $R_1$ and $R_2$ are more balanced. Thus, a population's receptor expression heterogeneity and variation are important factors to consider when modulating ligand affinity.

### Valency Enables Selectivity Based on Quantitative Differences in Receptor Abundance

Given the limitations of deriving selectivity from affinity changes, we next explored the effect of valency changes (Fig. {@fig:valency}). Our previous work has shown that this binding model can accurately predict the complicated multivalent binding process between IgG antibodies and Fcγ receptors [@pmid:29960887]. To further demonstrate our model's capacity to predict multivalent binding activity, we fit our model to a set of published experimental measurements wherein fluorescently labeled nanorings were assembled with specific numbers of binding units [@doi:10.1021/jacs.8b09198]. After fitting to determine the crosslinking coefficient of multivalent nanorings ($K_x^*$) and receptor affinity values ($K_a$), our model was able to accurately match the binding of nanorings with four or eight subunits to cells expressing a known abundance of receptor partners (Fig. S1a).

Multivalent ligand binding differs from monovalent binding in its nonlinear relationship with cellular receptor density, allowing multivalent ligands to potentially selectively target cells based on receptor abundance [@doi:10.1038/srep40098]. The effects of varying ligand valency are visualized in Fig. {@fig:valency}a. Here, the ligand only binds to receptor $R_1$, not receptor $R_2$, and the valency of the ligand is varied across subplots. It is shown that varying ligand valency results in a nonlinear relationship between receptor abundance and binding magnitude. For example, in the monovalent case, there are roughly the same amount of contour lines between $R_1^{lo}R_2^{lo}$ and $R_1^{med}R_2^{lo}$ as there are between $R_1^{med}R_2^{lo}$  and $R_1^{hi}R_2^{lo}$. However, in the tetravalent case, there are comparatively more contour lines between $R_1^{med}R_2^{lo}$ and $R_1^{hi}R_2^{lo}$, indicating that ligand bound is a nonlinear function of receptor expression when considering multivalent binding (Fig. {@fig:valency}a).

Selectivity derived by changing valency requires coordinate changes in affinity. As many previous studies have suggested, multi-valent ligands can demonstrate particular selectivity to target populations with high receptor abundances when their subunits have low affinity to the receptors [@doi:10.1021/cb6003788; @doi:10.1021/jacs.8b09198]. Comparing binding between cell populations predicted by our model exposed to ligands of variable affinities over a range of ligand valency demonstrates the interplay between these factors (Fig. {@fig:valency}b-d). Here, the ligand binding ratios between cell populations are shown for ligands of high, medium, and low affinities  for $R_1$ ($K_d$ of $1000$, $100$, and $10$ $\mathrm{nM}$), and the shaded areas indicate the variance caused by intrapopulation heterogeneity, determined by sigma point filter. The ligand binding ratio between $R_1^{hi}R_2^{lo}$ and $R_1^{med}R_2^{lo}$ is maximized by low affinity ligands, but requires greater valency to achieve peak binding selectivity when compared to ligands of greater affinity (Fig. {@fig:valency}b). A similar valency optimum for a given affinity is seen for binding selectivity between $R_1^{hi}R_2^{hi}$ and $R_1^{med}R_2^{med}$, and $R_1^{hi}R_2^{med}$ and $R_1^{med}R_2^{med}$ (Fig. {@fig:valency}c-d). Ligands with lower affinities achieve optimal binding with higher valencies and exhibit higher selectivity for cells expressing greater amounts of receptor.

For any specific multivalent ligand at equilibrium, it may bind to any amount of receptors no higher than its valency, which we will refer to as its binding degree. Our model allowed us to identify the mechanism of valency-mediated receptor abundance selectivity by examining the distribution of ligands bound at each degree to different cells for octavalent ligands (valency = 8). A cell expressing $10^4$ receptors displays similar amounts of binding at each degree for ligands with dissociation constants of $1000$, $100$, and $10 \mathrm{nM}$ (Fig. {@fig:valency}e). However, a cell expressing 10% as many receptors exhibits extremely low amounts of higher-degree binding for ligand complexes of low binding affinity (Fig. {@fig:valency}f). This effect arises due to the ligand's rate of dissociation being larger than its multivalent binding association rate at low receptor abundances (Fig. {@fig:valency}g). Multivalent ligands will undergo initial binding events at rates unaffected by receptor abundance, but low affinity and receptor density severely limit the possibility of secondary binding events. In contrast, cells with higher receptor abundance accumulate ligand bound by having them binding multivalently, as the forward rate of secondary binding events is greater than that of receptor-ligand disassociation (Fig. {@fig:valency}f). This effect allows multivalent ligands to achieve selective binding to cells based on their receptor abundances, which monovalent ligands, even with engineered affinities, are unable to do.

![**Valency provides selectivity based on receptor expression levels.** a) Heat/contour maps of multivalent ligands bound to cell populations given their expression profiles of two receptors. Multivalent ligand subunits bind to only $R_1$ with an dissociation constant of $100 \mathrm{nM}$, and do not bind to $R_2$. Complexes vary in valency from 1 to 16. Ligand concentration $L_0=1 \mathrm{nM}$; crosslinking constant $K_x^*=10^{-10}$. b-d) Ligand binding ratio between various cell populations for ligands of valency ranging from 1 to 16. The shaded areas indicate the variance of binding ratios caused by the intrapopulation heterogeneity and estimated by sigma point filter. b) Ligand bound ratio of $R_1^{hi}R_2^{lo}$ to $R_1^{med}R_2^{lo}$, c) $R_1^{hi}R_2^{hi}$ to $R_1^{med}R_2^{med}$, and d) $R_1^{hi}R_2^{med}$ to $R_1^{med}R_2^{med}$. e-f) Number of ligand bound to each possible number of receptors for cells exposed to octavalent ligand complexes composed of subunits with dissociation constants of $1000$, $100$, or $10 \mathrm{nM}$ for receptor 1. e) Number of octavalent complex bound at each degree for a cell with $10^4$ receptors. f) Number of octavalent complex bound at each degree for a cell with $10^3$ receptors. g) Ratio of forward to reverse binding rate for secondary binding events for multivalent ligands to cells expressing variable amounts of receptors.](figure3.svg){#fig:valency}

### Non-Overlapping Ligand Targeting Drives a Limited Selectivity Enhancement of Mixtures

While most therapies rely on the action of a single molecular species, mixtures may enhance selectivity through combinations of actions [@doi:10.1038/nbt.1549]. Furthermore, some biologics inevitably act as mixtures of species through heterogeneity in glycosylation and the presence of endogenous ligands [@DOI:10.1002/rcm.3330]. Therefore, it is important to understand how mixtures of complexes influence the overall response.

To evaluate the contribution of mixtures, we evaluated model-predicted binding while varying the composition between two distinct monovalent ligands with either $R_1$ or $R_2$-preferred binding (Fig. {@fig:mixture}a), while maintaining low level cross activity. The trend that arises is very similar to an additive combination of the single ligand cases. This pattern highlights a key limitation of using mixtures for selectivity: selectivity between two populations varies monotonically with the composition, so any mixture combination is no better than using the more specific ligand entirely (Fig. {@fig:mixture}b).

While mixture engineering fails to enhance binding selectivity between two cell populations, it is potentially beneficial when considering two or more off-target cell populations. More specifically, when a target population expresses two target receptors, but off-target populations express each receptor individually in high amounts, drug mixtures can offer enhanced selectivity. For example, when maximizing targeting to $R_1^{hi}R_2^{hi}$ over $R_1^{hi}R_2^{lo}$ and $R_1^{lo}R_2^{hi}$, which individually express high levels of the receptors found on the $R_1^{hi}R_2^{hi}$, we show that a uniform mixture of ligands with high affinity for receptor 1 and 2 provides a modest improvement in targeting selectivity (Fig. {@fig:mixture}c). However, even in these cases, the magnitude of selectivity enhancement is modest. Finally, although we only consider the amount of binding, ligands can have non-overlapping signaling effects even with identical amounts of binding. In these cases, the effect of combinations can be distinct from either individual ligand [@pmid:28886385; @pmid:29960887].

![**Ligand mixtures with non-overlapping responses can enhance selectivity.** a) Heat/contour maps of multivalent ligands bound to cell populations given their expression profiles of two receptors. A mixture of monovalent ligands is used, with ligand 1 binding to receptor 1 and 2 with dissociation constants of $1 \mathrm{\mu M}$ and $10 \mathrm{\mu M}$ respectively, and ligand 2 binding to receptors 1 and 2 with dissociation constants of $10 \mathrm{\mu M}$ and $1 \mathrm{\mu M}$ respectively. Ligand concentration $L_0=1 \mathrm{nM}$; crosslinking constant $K_x^*=10^{-10}$. b,c) Ratio of ligand bound to cell populations exposed to monovalent mixtures of ligand 1 and 2. The ratio of the target population to the single off target population with the greatest ligand bound is plotted. The shaded areas indicate the variance caused by intrapopulation heterogeneity, estimated by sigma point filter. b) Ligand bound ratio of $R_1^{hi}R_2^{lo}$ to $R_1^{med}R_2^{lo}$, c) $R_1^{hi}R_2^{hi}$ to $R_1^{hi}R_2^{lo}$ and $R_1^{lo}R_2^{hi}$.](figure4.svg){#fig:mixture}

### Heterovalent Bispecific Ligands Exhibit Unique Charateristics When Activated Fully Bound

Constructing multispecific drugs has become a promising new strategy for finer target cell specificity with the advancement of engineering techniques [@doi:10.1038/s41586-020-2168-1]. However, the number of possible configurations of multispecific drugs is combinatorially large and impossible to enumerate. Here, we use the heterovalent bispecific ligand as an example to explore the unique benefit of multispecificity distinct from any strategy analyzed before. Here, we compare bispecificity with a 50%-50% mixture of two monovalent ligands, and a 50%-50% mixture of two different homogeneous bivalent ligands (Fig. {@fig:bispecific}i). These two strategies both have some similarities to bispecific therapeutics. A bispecific ligand contains two different ligand monomers with equal proportion, similar to a 50%-50% monovalent mixture. By comparing bispecific with a 50%-50% monovalent mixture, we can elucidate the extra benefit of tethering these two monomers into one complex. A bispecific ligand is also by nature bivalent, and by comparing it with homogeneous bivalent drugs, we can understand how having two different subunits in the same complex modify the behavior of a drug. Together, we sought to identify the unique advantages afforded by heterovalent ligands.

We first applied the binding model to predict the amount of ligand bound in bispecific drugs (Fig. {@fig:bispecific}a), a 50%-50% mixture of two monomers (Fig. {@fig:bispecific}b), and a 50%-50% mixture of two different homogeneous bivalents (Fig. {@fig:bispecific}c), with the same set of parameters in ligand concentration and affinities. Surprisingly, the patterns of ligand binding in these three cases are almost exactly the same, and bispecificity appeared to offer no unique properties. However, many bispecifics only impart their therapeutic action when both of their subunits bind to a target population. For example, in the design of bispecific antibodies, it is common to require both subunits bound to be effective. When the first antigen-binding region of a bispecific antibody docks at an antigen, it will only make a loose connection between the target and the antibody, unable to induce a strong immune response, and only when both antigen-binding regions bind to their target will this bispecific antibody be fully effective [@doi:10.1080/19420862.2015.1062192; @pmid:30145356]. We investigated if bispecific ligands that have to be activated with both of their subunits binding to a target display any special cell population selectivity characteristics. 

To scrutinize the case for bispecific ligands where the target binding of both subunits is required, we extended our model to calculate the amount of ligand fully bound (Fig. {@fig:bispecific}i). For the heterovalent bispecific case, ligand fully bound will only account for the amount of ligand with both of their ligand monomers bound to a target. With the same set of parameters, the predictions made for bispecific fully bound show a very distinct pattern from general ligand bound (Fig. {@fig:bispecific}e). The contour plot of fully bound bispecific ligands has more concaved contour lines. For example, $R_1^{hi}R_2^{hi}$ has about the same level of general ligand bound as $R_1^{hi}R_2^{med}$ (Fig. {@fig:bispecific}a), but it has significantly more ligand fully bound than $R_1^{hi}R_2^{med}$ (Fig. {@fig:bispecific}e). This concavity of contour lines indicates that for bispecific complexes, double-positive cells bind more ligands fully than singly-positively expressed cells. This should be obvious since these two subunits of the bispecific complex prefer different receptors. The results indicate that bispecific ligands will only exhibit special characteristics when it requires both receptor binding subunits to bind a receptor.

The specific amount of fully bound ligands depends on the propensity of crosslinking, captured by the constant $K_x^*$ (Fig. {@fig:bispecific}i). Typically in biochemistry, when there is less steric hindrance among the subunits of a multivalent drug molecule (e.g. longer tether and smaller subunit size), or when there is local receptor clustering on the target cell, secondary binding will be easier to achieve, corresponding to larger $K_x^*$. We plotted the pattern of bispecific fully bound with $K_x^* = 10^{-10}$, $10^{-12}$, and $10^{-14}$ (Fig.{@fig:bispecific}d-f). In general, when $K_x^*$ is larger, the ligands are more capable of multimerization, and there will be more fully bound units. To demonstrate how this characteristic of fully bound bispecific ligand imparts cell population specificity, we compared the selectivities between some chosen target-to-off-target cell population pairs under bispecific drug versus another drug given a range of $K_x^*$ (Fig. {@fig:bispecific}g,h). These plotted numbers are the selectivities imparted by a bispecific drug divided by the selectivities from a drug mixture of either monovalent (Fig. {@fig:bispecific}g) or homogeneous bivalent (Fig. {@fig:bispecific}h). When these quotients are larger, it implies that fully bound bispecific ligand has greater advantages than its counterpart to impart selective binding for the chosen target cell populations. Figure {@fig:bispecific}g compares the selectivities under bispecific ligands versus a 50%-50% mixture of monovalent ligands. The results show that fully bound bispecific can grant better selective binding when $K_x^*$ is small enough. This is logical, since when $K_x^*$ is small and crosslinking is rarer, most ligands will bind monovalently, and fully bound bispecific will especially favor the cell populations with higher receptor expression. However, when we compare fully bound bispecific to fully bound homogeneous bivalent mixtures (Fig. {@fig:bispecific}h), the advantage of bispecific drugs is not obvious except for $R_1^{med}R_2^{hi}$ to $R_1^{hi}R_2^{med}$ selectivity. Given that we only account for fully bound ligand for both therapeutics, the effect demonstrated in Figure {@fig:bispecific}g no longer holds. Together, we showed that bispecific ligands only exhibit unique advantages in inducing selective binding when they are only effective when both of their subunits bind and crosslinking is more difficult.

![**Bispecific ligands exhibit unique effects when they can only be activated with both subunits bind.** a-c) Ordinary ligand bound are dominated by the initial binding of ligands, so it doesn't provide unique advantages to a) bispecific ligand comparing with b) a 50%-50% mixture of monovalent ligands and c) a 50%-50% mixture of bivalent ligands. Ligand concentration $L_0=10 \mathrm{nM}$; binding affinities $K_{d11}=100 \mathrm{nM}, K_{d22}=1 \mathrm{\mu M}, K_{d12}=K_{d21}=10 \mathrm{\mu M}$. d-f) The amount of fully bound bispecific ligands depends on the tendency of multimerization, capsuled by $K_x^*$. d) $K_x^* =10^{-10}$, e) $K_x^* =10^{-12}$, f) $K_x^* =10^{-14}$. g,h) Comparing bispecific selectivities with mixture selectivities, varies with $K_x^*$, the crosslinking constant. When the ratios are larger, bispecific ligands bind to target populations more specifically. g) bispecific selectivities divided by monovalent 50%-50% mixture selectivities, h) bispecific selectivities divided by a 50%-50% homo-bivalent mixture selectivities. i) Cartoons of bispecific ligands and ligand fully bound binding model.](figure5.svg){#fig:bispecific}

### Combining Strategies For Superior Selectivity

Each strategy described above provided selectivity benefits in distinct situations, suggesting that they might synergistically improve selectivity when combined. We explored this potential synergy using mathematical optimization to determine the ligand specifications which provided optimal selectivity for our theoretical cell populations. More specifically, we optimized the selectivity of a ligand for a particular population, while considering all other populations to be off-target. Our optimization allowed for ligand characteristics to vary within biologically plausible bounds; which characteristics we allowed to vary defined our examination of the efficacy of our various ligand engineering strategies. We examined optimizing affinity alone, mixture along with affinity, and valency along with affinity, and finally combined all three strategies (Fig. {@fig:combination}).

Optimizing a ligand for selectivity to $R_1^{lo}R_2^{hi}$ highlights a situation in which affinity imparts greater specificity, and optimal selectivity is achieved by combining affinity and valency modulations (Fig. {@fig:combination}a-f). Here selectivity is optimized by ligands with selective binding to receptor 2, and those with valencies allowing for selective binding to cells with higher abundances of receptor expression. One case contradictory to this trend is shown during the optimization for selectivity towards the $R_1^{hi}R_2^{hi}$ (Fig. {@fig:combination}g-l). While affinity engineering is unable to impart some small contributions to enhanced selectivity, significant improvement is only achieved when utilizing valency modulation techniques. A more difficult design problem is featured in the optimization of $R_1^{med}R_2^{med}$ (Fig. {@fig:mixture}m-r). Since it lies in the midst of the other populations in receptor expression space, any modulation of affinity, valency or combining it with mixture-based strategies seems ineffective. It should be noted that in all described cases, engineering the composition of a mixture of ligands is generaly ineffective for imparting selectivity when the ligand's design specifications are flexible, and is likely only efficacious when using ligands with static properties and considering multiple off-target populations.

Our results highlight that both in singular and combined strategies for therapeutic manipulation, the target and off-target populations dictate the optimal approach. It is also clear that combined approaches do offer synergies which can be harnessed, but that those are only emergent in particular therapeutic situations [@pmid:30145356].

![**Combinations of strategies provide superior selectivity.** a,g,m) Optimal selectivity levels (average ligand bound compared to average ligand bound by all other populations) achieved using various ligand engineering techniques. Ligand concentration $L_0=1 \mathrm{nM}$. Xo ligands are monovalent ligands with affinities of $1 \mathrm{\mu M}$ for both receptor 1 and 2. The dissociation constant was allowed to vary between $10 \mathrm{mM}$ and $0.1 \mathrm{nM}$ for both receptors using the "affinity" approach. Valency was allowed to vary from 1 to 16 for the "valency" approach in addition to affinities varying. Mixtures were assumed to be composed of two monovalent ligands, and affinities were allowed to vary in the "mixture" approach. The combined "all" approach allowed all of these quantities to vary simultaneously. The crosslinking constant $K_x^*$ was allowed to vary between $10^{-15}$ and $10^{-9}$ for all approaches. b-f,h-l,n-f) Heatmap of magnitude of ligand bound for ligand with optimized characteristics according to various ligand engineering strategies. Target population is shown in red. a-f) Pertains to optimal targeting of $R_1^{lo}R_2^{hi}$, g-l) pertains to optimal targeting of $R_1^{hi}R_2^{hi}$, and m-r) pertains to optimal targeting of $R_1^{med}R_2^{med}$.](figure6.svg){#fig:combination}

### Using Binding Competition to Invert Receptor Targeting

While the strategies above provided selectivity in many cases, we recognized that they are all limited to a positive relationship between receptor abundance and binding. Therefore, we wondered if binding competition with a receptor antagonist, or "dead ligand", could invert this relationship.

To investigate the effect of ligand competition with an antagonist, we modeled mixtures of ligands, but only quantified the amount of binding for the active ligand. Here we chose to only consider a monovalent agonist and tetravalent antagonist (Fig. {@fig:deadLig}). We found that combinations of monovalent agonistic ligands and multivalent antagonistic ligands were able to uniquely target populations expressing small or intermediate amounts of receptors, which is demonstrated when comparing ligand binding ratios between $R_1^{med}R_2^{lo}$ to $R_1^{hi}R_2^{lo}$ (Fig. {@fig:deadLig}a). Here, a nearly tenfold increase in selectivity can be granted to monovalent agonists when combined with a tetravalent antagonist. In this case, there are greater quantities of agonist bound to $R_1^{med}R_2^{lo}$ than $R_1^{hi}R_2^{lo}$ (Fig. {@fig:deadLig}b). This is striking, as $R_1^{med}R_2^{lo}$ expresses either as many or fewer abundances of receptors one and two when compared to $R_1^{hi}R_2^{lo}$. This phenomenon, which could not be achieved without multivalent antagonists, occurs due to the preferential binding of multivalent antagonists to populations expressing higher abundances of receptors (Fig. {@fig:deadLig}c, {@fig:valency}e). Thus, in cases where previously discussed ligand engineering strategies and approaches fail to achieve selective binding to cells expressing smaller or similar amounts of receptors to off-target populations, combinations of agonistic and antagonistic ligands may provide unique benefits.

![**Mixtures of receptor agonists and antagonists allow for unique population targeting activity.** Ligand concentration $L_0=1 \mathrm{nM}$. a) Selectivity for $R_1^{med}R_2^{lo}$ against $R_1^{hi}R_2^{lo}$ when exposed to a tetravalent "dead ligand" antagonist with varying affinities for receptors 1 and 2, and a monovalent therapeutic receptor agonists with affinities optimized for selectivity. Only amount of agonist bound is considered in determination of optimal selectivity. b-d) Heatmap of agonist (b), and antagonist (c) ligand bound for antagonist and agonist ligand combination shown to impart greatest selectivity improvement in (a). d) Heatmap of agonist bound in b,c when no antagonist is present.](figure7.svg){#fig:deadLig}

## Discussion

<!-- Summary. -->

Here, we developed and employed a multivalent, multi-ligand, multi-receptor binding model to explore the effectiveness of various ligand engineering strategies for population-selective binding (Fig. {@fig:model}). Using a representative set of theoretical cell populations defined by their distinct expression of two receptors, we examined the efficacy of several potential ligand engineering strategies, including changes to affinity, ligand valency, mixtures of species, multi-specificity, and these in combination.

Each strategy’s contribution can be summarized by general patterns. We found that affinity changes were most effective when the target and off-target populations expressed distinct combinations of receptors (Fig. {@fig:affinity}). Binding selectivity was enhanced by increasing the affinity of the ligand for those receptors that are more abundantly expressed on the target cells and decreasing its affinity for those that are not. Selectivity depended upon large differences in receptor expression levels between on- and off-target populations. When target and off-target populations expressed the same pattern of receptors and only differed in receptor abundances, modulations in valency, but not affinity, were effective (Fig. {@fig:valency}). A key determinant of valency’s effectiveness was the secondary binding and unbinding rate which is dependent on both the kinetics of the receptor-ligand interaction and receptor abundance. Ligand mixtures were mostly ineffective for imparting binding selectivity, and only had modest benefits when considering three or more off-target populations (Fig. {@fig:mixture}). Heterovalent bispecific ligands only showed unique advantages over mixtures of monovalent ligands or bivalent ligands when we restricted our binding quantity to be both subunits bound (Fig. {@fig:bispecific}). They prefer target populations that have high expression of both receptors over those with high expression of only a single receptor, with the preference for secondary binding as the key determinant for selectivity. We found that, while a single ligand engineering strategy dominated in its contributions to cell type selectivity, synergies between these strategies existed in some cases to derive even greater specificity (Fig. {@fig:combination}). Finally, we found that combinations of monovalent therapeutic ligands with multivalent antagonistic ligands allow for selection of target populations based on their lack of receptor expression (Fig. {@fig:deadLig}).

<!-- No strategies for NOT relationships. -->

While our multivalent binding model identified strategies for selective targeting in many cases, it also identified situations for which selective binding is challenging. For example, selectively targeting populations based on their absence of receptor expression remains challenging. While we computationally show the potential of using multivalent antagonists with monovalent agonists, implementing this may be challenging. In cases where a target population expresses fewer receptors of any kind than an off-target population, our analysis suggests that targeting other receptors should be considered. However, in cases where target populations express more of any type of receptor than an off-target population, we show that one or more of our formulated ligand engineering strategies can be employed to improve binding selectivity. While we expect the same patterns to apply with greater than two receptors, certain emergent behaviors may exist with tri-specific and more complex ligand binding.

<!-- A number of strategies are already employed. -->

A few of the strategies that we explored are already utilized in existing engineered therapies. For example, affinity changes to the cytokine IL-2 have been used to bias its effects towards either effector or regulatory immune populations [@doi:10.1158/2159-8290.CD-18-1495; @pmid:30446251]. Varying the valency of tumor-targeted antibodies leads to selective cell clearance based upon the levels of expressed antigen [@doi:10.1038/srep40098]. Manipulating of the affinities of the fibronectin domains on octovalent nanorings was shown to enhance the selectivity of binding to cancerous cells displaying relatively higher densities of fibronectin receptors compared to native tissue [@doi:10.1021/jacs.8b09198]. The tendency of low-affinity, multivalent interactions to target cells expressing high receptor abundances was also described in a study describing the selectivity of multivalent antibody binding to tumor cells bound by a bispecific therapeutic ligand [@doi:10.1021/cb6003788]. These examples lend support to the accuracy of our model. At the same time, recognizing these previously described ligand engineering approaches as separable strategies provides clearer guidance for future engineering.

<!-- Still need to implement others. -->

Some of the optimization strategies described here have not been exploited in part due to the complexity of real biological applications. First, some strategies may not be biochemically practical. For example, the manipulating ligand affinity requires intricate protein engineering. Potential dynamic changes in the receptor expression profile of a target population also complicate the matter. It is well documented that cancer cells can escape therapeutic targeting by upregulating [@doi:10.1016/j.devcel.2019.07.010; @doi:10.1158/0008-5472.CAN-13-0602] or downregulating [@pmid:14534734] the expression of certain receptors. In this case, both the current and potential abundance of each receptor must be considered. While this work does not intend to solve every aspect of these issues, we propose that using a computational binding model can analyze these strategies quantitatively and collectively from a mechanistic perspective. Even when the absolute mathematical optimum cannot be achieved biochemically, our analyses provide good guidance to what is attainable and how to approach the optimum, accounting for implementation feasibility and facilitating the implementation of strategy combinations.

<!-- Impressive range of logic can be built without cells involved. -->

In many therapeutic applications where selective engagement of target cell populations is an important performance metric, such as the treatment of cancer, cellular therapies are becoming increasingly popular [@doi:10.1146/annurev-immunol-042718-041407]. Human engineered chimeric antigen receptor (CAR) T cells have enhanced the potential to selectively recognize and attack malignant tissues [@doi:10.1016/j.ymthe.2017.06.012]. These technologies bypass ligand-receptor binding restrictions by allowing recognition in signaling response. However, we have shown here that selectivity can often be attained with relatively simple therapeutic ligands. This study lays the framework for how ligand engineering can be directed using computational modeling. It should be noted that the application of this logic is reliant on knowledge of the target and off-target cell population receptor expression levels. Future application of the ligand binding logic described in this study could be guided using high-throughput single-cell profiling techniques, such as RNA-seq or high-parameter flow cytometry. A computational tool that could directly translate such datasets into ligand design criteria may represent a potential avenue for the translation of our analyses into a more broadly applicable ligand engineering tool.



## Materials and Methods

### Data and Software Availability

All analysis was implemented in Python v3.8, and can be found at <https://github.com/meyer-lab/cell-selective-ligands>.

### Generalized multi-ligand, multi-receptor multivalent binding model

To model multivalent ligand complexes, we extended our previous binding model to the multi-ligand case [@pmid:29960887]. We define $N_L$ as the number of distinct monomer ligands, $N_R$ the number of distinct receptors, and the association constant of monovalent binding between ligand $i$ and receptor $j$ as $K_{a,ij}$. Multivalent binding interactions after the initial interaction have an association constant of $K_x^* K_{a,ij}$, proportional to their corresponding monovalent affinity. The concentration of complexes is $L_0$, and the complexes consist of random ligand monomer assortments according to their relative proportion. The number of ligand complexes in the solution is usually much greater than that of the receptors, so we assume binding does not deplete the ligand concentration. The proportion of ligand $i$ in all monomers is $C_i$. By this setup, we know $\sum_{i=1}^{N_L} C_i = 1$. $R_{\mathrm{tot},i}$ is the total number of receptor $i$ expressed on the cell surface, and $R_{\mathrm{eq},i}$ the number of unbound receptors $i$ on a cell at the equilibrium state during the ligand complex-receptor interaction.

The binding configuration at the equilibrium state between an individual complex and a cell expressing various receptors can be described as a vector $\mathbf{q} = (q_{1,0}, q_{1,1}, ..., q_{1,N_R}, q_{2,0},..., q_{2,N_R},q_{3,0},..., q_{N_L, N_R})$ of length $N_L(N_R+1)$, where $q_{i,j}$ is the number of ligand $i$ bound to receptor $j$, and $q_{i,0}$ is the number of unbound ligand $i$ on that complex in this configuration. The sum of elements in $\mathbf{q}$ is equal to $f$ , the effective avidity. For all $i$ in $\{1,2,..., N_L\}$, let $φ_{i,j} = R_{\mathrm{eq},j} K_{a,ij} K_x^* C_i$ when $j$ is in $\{1,2,...,N_R\}$, and $φ_{i,0} = C_i$. The relative amount of complexes in the configuration described by $\mathbf{q}$ at equilibrium is


$$v_{\mathbf{q},eq} = \binom{f}{\mathbf{q}} \frac{L_0}{K_x^* } \prod_{i=1\\ j=0}^{i=N_L\\ j=N_R}{{φ_{ij}}^{q_{ij}}},$$


with $\binom{f}{\mathbf{q}}$ being the multinomial coefficient. Then the total relative amount of bound receptor type $n$ at equilibrium is


$$ R_{\mathrm{bound},n} = \frac{L_0 f}{K_x^* } \sum_{m=0}^{N_L}φ_{mn} \left(\sum_{i=1\\ j=0}^{i=N_L\\ j=N_R}{{φ_{ij}}^{q_{ij}}}\right)^{f-1} .$$


By conservation of mass, we know that $R_{\mathrm{tot},n} = R_{\mathrm{eq},n} + R_{\mathrm{bound},n}$ for each receptor type $n$, while $R_{\mathrm{bound},n}$ is a function of $R_{\mathrm{eq},n}$. Therefore, each $R_{\mathrm{eq},n}$ can be solved numerically using $R_{\mathrm{tot},n}$. Similarly, the total relative amount of complexes bound to at least one receptor on the cell is


$$ L_{\mathrm{bound}} = \frac{L_0}{K_x^* } \left [\left(\sum_{i=1\\ j=0}^{i=N_L\\ j=N_R}{{φ_{ij}}^{q_{ij}}}\right)^f -1 \right] .$$


### Generalized multivalent binding model with defined complexes

When complexes are engineered and ligands are not randomly sorted into multivalent complexes, such as with the Fabs of bispecific antibodies, the proportions of each kind of complex become exogenous variables and are no longer decided by the monomer composition $C_i$'s. The monomer composition of a ligand complex can be represented by a vector $\mathbf{θ} = (θ_1, θ_2, ..., θ_{N_L})$, where each $θ_i$ is the number of monomer ligand type $i$ on that complex. Let $C_{\mathbf{θ}}$ be the proportion of the $\mathbf{θ}$ complexes in all ligand complexes, and $Θ$ be the set of all possible $\mathbf{θ}$'s. We have $\sum_{\mathbf{θ} \in Θ} C_{\mathbf{θ}} = 1$.

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

We used the SciPy function `scipy.optimize.minimize` to combine several strategies and achieve optimal selectivity [@doi:10.1038/s41592-019-0686-2]. Unless specified otherwise, the initial values for optimization were $10^{-12}$ for crosslinking coefficient $K_x^*$, 1 for valency $f$, 100% ligand 1 for mixture composition, and $1 \mathrm{\mu M}$ for the affinity dissociation constants. The boundaries were $10^{-15} \sim 10^{-9}$ for $K_x^*$, 1–16 for $f$, 0–100% ligand 1 for mixture composition, and $10 \mathrm{mM} \sim 0.1 \mathrm{nM}$ for the affinity dissociation constants.

### Sigma point filter

To consider the intrapopulation variance of a cell population in the optimization, we implemented sigma point filter [@wikidata:Q99353631], a computationally efficient method to approximate the variance proprogated through an ordinary differential equation-based model.

### Reimplementation of Csizmar et al.

To validate our model, we recapitulated multivalent binding data from Csizmar et al. using our multivalent binding model (Fig. S1) [@doi:10.1021/jacs.8b09198]. Here, fluorescently labeled nanorings displaying 1, 2, 4, and 8 fibronectin clones, which bind to epithelial cell adhesion molecule (EpCAM) antigens, were fabricated. Binding activity of nanorings displaying both high (C5) and low (B22) affinity EpCAM binding domains was measured. Binding to an EpCAM^high^ ($3.3 \times 10^6$ antigens/cell) population was measured using flow cytometry. We used nonlinear least squares optimization, as described above, to fit our multivalent binding model to the binding data, using a unique scaling factor for each fibronectin clone to convert between measured fluorescent intensity and magnitude of ligand bound. We allowed affinity of the fibronectin clones to vary during optimization.


## Supplementary Figures

![**Our model is able to recapitulate experimental multivalent binding activity.** a) Experimental vs. predicted fluorescent units for MCF-7 cells expressing $3.8 \times 10^{6}$ fibronectin receptors per cell bound to nanorings expressing one, two, four or eight fibronectin binding domains at concentrations of 0.16–500 nM. C5 and B22 are high and low affinity fibronectin binding domains respectively. Using nonlinear least-squares regression, the crosslinking coeffient was found to be $1.11 \times 10^{-12}$; the association constants for C5 and B22 were found to be $5.78 \times 10^{-5}$, $3.08 \times 10^{-12}$ respectively, and ligand to fluorescent conversion factor for C5 and B22 were found to be $5.0 \times 10^{-2}$ and $3.2 \times 10^{-2}$ respectively. b) Predicted fluorescent values for MDA cells expressing $5.2 \times 10^{4}$, SK cells expressing $2.2 \times 10^{5}$, LNCaP cells expressing $2.8 \times 10^{6}$, and MCF-7 cells expressing $3.8 \times 10^{6}$ receptors per cell as described in Csizmar et al. [@doi:10.1021/jacs.8b09198]. Fluorescence was predicted for interactions with octa- and tetravalent ligands expressing either C5 or B22 fibronecting binding domains. c) Predicted binding ratios of MCF-7 cells to MDA cells when bound with octa- and tetravalent ligands expressing either C5 or B22 fibronecting binding domains.](figureS1.svg){#fig:Csizmar}


## Acknowledgements

This work was supported by NIH U01-AI148119 to A.S.M. **Competing financial interests:** The authors declare no competing financial interests.



## Author contributions statement

A.S.M. concieved of the work. All authors implemented the analysis and wrote the paper.


## References {.page_break_before}

<!-- Explicitly insert bibliography here -->
<div id="refs"></div>
