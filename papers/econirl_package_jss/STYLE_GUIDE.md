# Style guide for the econirl JSS paper

This guide distills the conventions of four canonical software papers in the same tradition. Use it as the prose checklist when drafting or revising any section of the paper.

## Primary exemplar

**Conlon and Gortmaker (2020), "Best Practices for Differentiated Products Demand Estimation with PyBLP."** This is the paper closest in shape to what we are writing. It is a long-form software paper that bridges a methodological literature with a Python package, includes original methodological contributions alongside the implementation, and targets a journal with conventions close to JSS. Mimic its rhythm and register.

## Secondary exemplars

- **Rosseel (2012), "lavaan: An R Package for Structural Equation Modeling"** in JSS. The template for code-listing format and the "Why do we need X" framing.
- **Gleave et al. (2022), "imitation: Clean Imitation Learning Implementations."** The template for the related-software comparison table.
- **Raffin et al. (2021), "Stable-Baselines3."** The template for the hero teaser code block.

## Ten rules to write by

1. **Open the abstract with the substantive problem, not the package name.** PyBLP starts "Differentiated products demand systems are a workhorse for understanding the price effects of mergers, the value of new goods, and the contribution of products to seller networks." The package is named only in the third sentence. Do the same. Open with why decoding behavior into preferences and beliefs matters for industrial organization, labor economics, urban economics, and AI alignment.

2. **End the abstract with one install line and one URL.** Both PyBLP and imitation close their abstracts with `pip install <pkg>` and the documentation URL. Match this.

3. **Put models before software.** PyBLP devotes six pages to the BLP model and notation before introducing the package architecture. lavaan devotes Section 3 to model syntax before any code. The reader needs to know what is being abstracted before they can appreciate the abstraction. Move the API discussion after the methods.

4. **Have a "Why do we need econirl?" section.** lavaan dedicates Section 2 to this question with a numbered list of three target audiences. State plainly that no existing package covers both NFXP and AIRL with shared inference. Three target audiences for econirl are structural econometricians who want a path to neural rewards, IRL researchers who want standard errors and identification diagnostics, and applied empiricists who want to fit both families on the same panel without reimplementing either.

5. **Use a comparison table for related software.** imitation and SB3 both pair a prose paragraph with a feature table. Rows are estimators and infrastructure features. Columns are competing libraries. Put this in its own section after the methods, not buried in the introduction.

6. **Worked examples follow the lavaan inline-session template.** Show the prompt, the input, the printed output, then prose that interprets the output. Example:

   ```
   >>> from econirl.datasets import load_rust_bus
   >>> panel = load_rust_bus()
   >>> result = NFXP(discount_factor=0.9999).estimate(panel)
   >>> print(result.summary())
                     coef    std err         z      P>|z|
   theta_1       0.001231   0.000087     14.16     0.000
   RC            3.011476   0.245312     12.27     0.000
   ```

   Then interpret. Do not paste full scripts as captioned figures.

7. **Use first person plural.** "We introduce econirl." "Our package implements." "We benchmark." All four exemplars use this register freely. Avoid the passive third person ("the package is presented").

8. **Bold package names. Code-font class and function names. Italicize terms of art on first use.** "We provide **econirl**, a Python package for the estimation of dynamic discrete choice models." "The `Estimator.estimate()` method." "We project the neural reward onto a *sieve basis*."

9. **Three registers: humble on shared methodology, declarative on novel contribution, comparative on prior software.** Borrow PyBLP's "our results generally coincide with and build on the existing literature" alongside its "we struggle to replicate some of the difficulties found in the previous literature." Borrow imitation's "By contrast, prior libraries typically support only a handful of algorithms, are no longer actively maintained, and are built on top of deprecated frameworks."

10. **Write long expository paragraphs, not bullet lists.** PyBLP and lavaan never use bullets in body prose. Five to eight sentences per paragraph is the right rhythm. If you find yourself writing a bullet list outside of an enumerated list of target audiences, convert it to prose.

## Section ordering for econirl

The recommended ordering, restructured from the original plan:

1. **Introduction.** Substantive problem, narrative of the convergence between structural econometrics and inverse reinforcement learning, statement of the fragmentation problem, one announcement paragraph for econirl, one five-line teaser code block. No related work yet. No contribution list yet.
2. **Why econirl.** Three target audiences in one paragraph each. Closes with three differentiators in prose.
3. **Models and methods.** Unified notation table at the top. Each estimator family gets a subsection with display equations. Taxonomy table at the end.
4. **Related software.** Standalone section with comparison table.
5. **Software design.** `Estimator` protocol, `EstimationSummary`, inference layer.
6. **Illustrations.** Three worked examples in inline-session format, real named datasets, benchmark table at the end.
7. **Computational details.**
8. **Summary and roadmap.**

The three swaps from the original outline are pulling related work out of the introduction into its own section, inserting "Why econirl" before the methods, and putting related work after the methods so the reader can evaluate the comparison fairly.

## Phrases to memorize

These templates appear repeatedly in the exemplars and should be reused without apology.

- "The rest of this paper is organized as follows."
- "Why do we need X."
- "A first example: Y."
- "The main disadvantage of X is."
- "Despite its popularity, this literature lacks."
- "A key advantage of X is."
- "By contrast."
- "We find that."
- "Our objective is."

## Things to avoid

- Em dashes, semicolons, colons, plus signs, equals signs, or brackets in prose. These belong in code and math.
- Hedging phrases like "we believe," "it should be noted," "arguably." Make the claim or drop it.
- "Cutting-edge," "state-of-the-art," "novel" as self-applied adjectives. Let the comparison table make the claim.
- Bullet lists in body prose. Convert to enumerated lists only for the three target audiences.
- Restating what a table or figure already shows. Prose interprets, not summarizes.
- Sentences that overload multiple ideas with semicolons. Say one thing per sentence.

## Reference checklist before submission

Before sending the manuscript, verify each of the following.

1. The abstract opens with the substantive problem and closes with the install line and URL.
2. Section 2 names three target audiences in three paragraphs.
3. Section 3 has a unified notation block before any estimator subsection.
4. Section 4 has a comparison table with at least four competing libraries.
5. Section 6 examples use the inline-session prompt-and-output format, not script blocks as captioned figures.
6. No prose paragraph contains an em dash, a semicolon, or a colon outside of math or code.
7. No paragraph in the body uses a bullet list.
8. Every numbered table and figure is followed by interpretive prose, not a restatement.
9. Every estimator named in the paper appears in both the taxonomy table and the benchmark table.
10. Every numerical claim in the paper traces to a script in `replication.md`.
