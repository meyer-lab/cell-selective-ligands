## Results

### Model System For Exploring The Contributing Factors To Cell Selectivity

![**A model system for exploring the contributing factors to cell selectivity.** A) Experimentally measured IL-2Rα and IL2Rβ abundances of a panel of PBMC-derived subpopulations. Relative receptor abundances are a key factor in cell-selective ligand engineering. B) Receptor abundances of a panel of theoretical cell populations. Strategies for selectively binding each example population relative to another are explored. D) Association constants ($K_a$) of several γ~c~ cytokine and FcG receptor-ligand complexes. Both γ~c~ cytokines and FcG ligands are often engineered for enhanced cell-type selectivity.](./output/figure1.svg){#fig:model}

### Affinity Provides Selectivity To Cell Populations With Unique Receptor Expression

Text.

When the cell population of interest expresses a higher level of a certain receptor than the others to the same ligand, enhancing the affinity of this receptor or reducing the affinity of competing receptors are obvious strategies to increase ligand bound. As demonstrated in Fig. X, cell type 3 expresses a high amount of receptor 1, while type 4 receptor 2. When the affinity of receptor 1 to the ligand is enhanced, the amount of ligand binds to cell type 3 increases significantly compared to type 4 (Fig X-line). As a general trend, all cell types with high expression of receptor 1 see an increased ligand bound. The contour plot on the background shows the trend more intuitively: when the affinity of receptor 1 increases (from the subplots on the left to those on the right), the contour lines shift inward and cluster denser, indicating higher rates of growth on the direction of more abundant receptor 1.

But commonly in biology, multiple cell populations, including those we aim for and against, express the same set of receptors and only differ in amounts, such as cell types 5 and 6 shown (Fig). In this case, the efficacy of enhancing a certain receptor’s affinity still holds, though less prominently. It is worthwhile to identify which receptors these two cell populations express the most divergently to maximize the selectivity in binding.

When a cell population expresses higher levels of effective receptors of every kind, such as type 7 in Fig X, it will always have a higher amount of bound ligand, unless there is any allosteric effect or other conformation changes beyond the scope of our discussion. When this population is indeed the target, some optimization can still be performed to maximize the binding selectivity to it over other populations. Fig X-line demonstrates the ratio of binding between cell type 7 and type 8 which has with lower amounts of every receptor, given various affinities. Such optimization is not futile in drug design, especially when undevised binding leads to detrimental side effects (ref).

The variations of receptor expressions within a cell population can induce noises in selectivity. For example, in Fig. X, receptor abundances of type 8 have high variance. The eclipse representing its range of receptor abundances rides through multiple contour lines, indicating how wide the bound ligand values span. If we choose to maneuver the affinity of receptor 1 to attain more selective binding to type 3 over type 8, it can be expected that the binding ratio will exhibit higher fluctuation (Fig X-line).

![**Affinity provides selectivity to cell populations with unique receptor expression.** A) Receptor abundances of a panel of theoretical cell populations. B-F) Heatmap of binding ratio of cell populations exposed to a monovalent ligand with association constants to receptor 1 and 2 ranging from 10^4^ to 10^9^, at a concentration of 10 nM. B-E) Heatmaps show that engineering affinity is effective when target population has a greater relative abundance of one receptor to the off-target population, and that binding ratio ratio is largest when ligand is engineered to selectively bind that receptor. The maximum binding ratio achieved using this strategy is the ratio of the receptor on the target and off-target population. E) Heatmap shows that when relative abundance of both receptors is uniform between target and off-target population is uniform, engineering affinity is ineffective.](./output/figure2.svg){#fig:affinity}

### Valency Enables Selectivity To Cell Populations Based On Receptor Expression Levels

While affinity modulation techniques are effective for the selective targeting of cell populations expressing unique or mutated receptors, they remain largely inadequate for the targeting of cell populations expressing receptor profiles similar to those of off-target populations. In such cases, therapeutic ligands solely engineered with altered receptor binding activity are known to suffer from significant off-target toxicities, as monovalent ligands fail to distinguish between populations with varying receptor abundances, binding each with equal avidity [@doi:10.1021/cb6003788]. Valency engineering offers a promising avenue for distinguishing between such divergent populations, as multivalent ligand binding avidity is known to vary as a function of cellular receptor abundance, as well as ligand concentration and binding affinity [@doi:10.1021/jacs.8b09198]. These characteristics governing multivalent binding has been leveraged for effective targeting of cancer cells known to overexpress receptors shared with native tissue, but holistic understanding of the binding mechanisms at play has remained elusive [@doi:10.1016/j.ijpharm.2007.07.040]. We utilized our binding model to explore the interplay between these factors, and to define engineering specifications for cell-type selective multivalent ligands.

Example: [@doi:10.1038/srep40098]

![**Valency provides selectivity based on receptor expression levels.** A) Receptor abundances of a panel of theoretical cell populations. B) Fraction of receptors bound by a high and low affinity ligand (($K_a$) = 10^8^, 10^7^ respectively) in four valencies over a range of receptors/cell. Fraction bound is constant for monovalent ligand, and increases at higher receptor counts for ligands of higher valencies. C) Fraction of receptors bound by a tetravalent ligand with a association constant of 10^8^ at a variety of concentrations over a range of receptors/cell. Values coalesce at higher receptor counts. D-I) Binding ratio between target and off-target populations at a range of valencies for ligands at pairs of high, medium, and low concentrations and affinities to receptor 1 and 2 (10 nM, 1 nM, and 0.1 nM, and 10^6^, 10^7^, and 10^8^ respectively). D-F) Varying valency is an effective strategy for target and off-target population pairs with large receptor abundance discrepancies, especially using ligands with low affinities at low concentrations. G-I) Binding ratios between population pairs with similar receptor counts fails to vary with valency of ligand.](./output/figure3.svg){#fig:valency}

### Non-Overlapping Ligand Targeting Drives Enhanced Selectivity Of Mixtures

Text.

Example: [@pmid:28886385]

![**Ligand mixtures with non-overlapping responses can enhance selectivity.** A) Receptor abundances of a panel of theoretical cell populations. B-F) Binding ratio between target and off-target populations exposed to a mixture of two monovalent ligands at a concentration of 10 nM. Ligand 1 binds with high affinity (($K_a$) = 10^9^) for receptor 1 and low affinity to receptor 2 (($K_a$) = 10^2^), while ligand 2 binds with high affinity to receptor two, and low affinity to receptor 1. B) Binding ratio between population 3 and 2. Mixtures cannot enhance selectivity between a target population and a single off-target population. C-F) Binding ratios between a target and multiple off-target populations. The minimum binding ratio between these pairs is reported. With multiple off-target populations, mixtures can achieve superior cell-type selectivity to singular ligand compositions.](./output/figure4.svg){#fig:mixture}

### Combining Strategies For Superior Selectivity

Example: [@pmid:30145356]

Text.

![**Combinations of strategies provide superior selectivity.**](./output/figure5.svg){#fig:combination}
