"""Plain-language definitions of the terms EDM-ARS results use.

``edmars explain AUC`` prints one of these. They are written for an
education researcher who has not used the method before: what the number
or word means, how to read it, and one caution. They are deliberately
short; each ends by pointing at what to check in the paper itself.
"""

from __future__ import annotations

import difflib
import re

#: Canonical term -> definition. Keys are lower case.
TERMS: dict[str, str] = {
    "auc": (
        "AUC (area under the ROC curve) measures how well a prediction model "
        "separates two groups, for example students who did and did not enrol "
        "in college. It is the chance that the model ranks a randomly chosen "
        "student from the first group above one from the second. 0.5 is a coin "
        "flip, 1.0 is perfect; in education data 0.70-0.80 is typical and above "
        "0.95 is suspicious (it often means information about the outcome leaked "
        "into the predictors). AUC says nothing about whether predicted "
        "probabilities are accurate: see 'calibration'."
    ),
    "rmse": (
        "RMSE (root mean squared error) is the typical size of a prediction "
        "error for a numeric outcome such as a test score or GPA, in the "
        "outcome's own units. An RMSE of 0.6 GPA points means predictions are "
        "usually off by about that much, with large misses weighted more. Lower "
        "is better. Compare it with the outcome's standard deviation: a model "
        "whose RMSE is close to the SD predicts little better than the average."
    ),
    "r-squared": (
        "R-squared is the share of the differences between students' outcomes "
        "that a model accounts for, from 0 (none) to 1 (all). On held-out test "
        "data it can even be negative, meaning the model does worse than "
        "predicting the average for everyone. Education outcomes are shaped by "
        "much that surveys do not measure, so values of 0.2-0.5 are common and "
        "not a sign of failure."
    ),
    "confidence interval": (
        "A confidence interval (CI) is a range of plausible values for a "
        "number estimated from a sample, such as [0.76, 0.80] around an AUC of "
        "0.78. A 95% CI is built by a method that captures the true value in 95% "
        "of repeated samples. Wide intervals mean the data cannot pin the number "
        "down; when two models' intervals overlap heavily, do not claim one is "
        "better. EDM-ARS usually computes them by bootstrapping (re-sampling the "
        "test set many times)."
    ),
    "shap": (
        "SHAP values show how much each predictor pushed one student's "
        "prediction up or down compared with the average prediction. Averaging "
        "their absolute size across students ranks the predictors by influence "
        "on the MODEL. They describe the model, not the world: a high SHAP value "
        "is not evidence that changing that predictor would change the outcome, "
        "and correlated predictors can share or swap credit."
    ),
    "calibration": (
        "Calibration asks whether predicted probabilities can be taken at face "
        "value: of the students given a 30% chance, did about 30% actually have "
        "the outcome? A model can rank students well (high AUC) and still be "
        "badly calibrated. Calibration plots and the Brier score check this. It "
        "matters whenever a probability, not just a ranking, is reported or used."
    ),
    "subgroup fairness": (
        "Subgroup fairness checks whether a model works equally well for "
        "different groups of students, for example by sex, race/ethnicity or "
        "family income. EDM-ARS reports accuracy measures separately for each "
        "group and flags large gaps. A model that is accurate on average can "
        "still be much worse for one group, which is one reason these models "
        "must not be used for decisions about individual students."
    ),
    "ate": (
        "The ATE (average treatment effect) is how much an outcome would change, "
        "on average across the population studied, if everyone received a "
        "'treatment' (a course, a program, an experience) compared with if no "
        "one did. With survey data it is estimated by comparing similar treated "
        "and untreated students, which is only valid if the variables used to "
        "judge 'similar' capture every important reason students differ in who "
        "got the treatment. Read the paper's assumptions and sensitivity checks."
    ),
    "propensity score": (
        "A propensity score is each student's estimated probability of receiving "
        "the treatment, given their measured background. Students with the same "
        "score are comparable on those measured variables, so treated and "
        "untreated students can be matched or weighted by it. It cannot fix "
        "differences in things that were not measured. Scores very close to 0 or "
        "1 mean some students have no comparable counterparts (poor 'overlap')."
    ),
    "matching": (
        "Matching pairs each treated student with one or more untreated students "
        "who look similar on measured background (often on the propensity "
        "score), then compares their outcomes. Check the balance table: after "
        "matching, the groups should differ little on every covariate (a "
        "standardized difference under 0.1 is a common rule of thumb). Students "
        "without a good match are dropped, which can change who the estimate "
        "describes."
    ),
    "ipw": (
        "IPW (inverse probability weighting) re-weights students by the inverse "
        "of their chance of getting the treatment they actually got, so the "
        "weighted treated and untreated groups resemble the whole population. "
        "Students with extreme propensity scores get huge weights that make "
        "estimates unstable, so weights are usually trimmed or stabilized; the "
        "paper should say how."
    ),
    "doubly robust": (
        "A doubly robust estimator (such as AIPW) combines a model of who gets "
        "the treatment with a model of the outcome. The estimate stays "
        "consistent if EITHER model is right, which makes it a safer default "
        "than either alone. It is not robust to unmeasured confounding: if an "
        "important reason for getting the treatment was never measured, both "
        "models share that blind spot."
    ),
    "causal forest": (
        "A causal forest is a machine-learning method (a forest of decision "
        "trees built for treatment effects) that estimates how a treatment's "
        "effect differs from student to student based on their characteristics. "
        "Individual estimates are noisy; the useful outputs are group-level "
        "patterns and tests of whether effects really vary. It rests on the same "
        "'no unmeasured confounding' assumption as other survey-based methods."
    ),
    "treatment rule": (
        "A treatment rule (individualized treatment rule, ITR) says which "
        "students are predicted to benefit from a treatment and should receive "
        "it, based on their characteristics. EDM-ARS compares the estimated "
        "average outcome under the learned rule with treating everyone or no "
        "one. Such a rule is a research finding about the data, not a policy "
        "ready to apply to real students."
    ),
    "difference-in-differences": (
        "Difference-in-differences compares how a gap (or an outcome) changed "
        "for one group between two times or cohorts with how it changed for a "
        "comparison group. In EDM-ARS's 'gap-in-gaps' design it compares how an "
        "achievement or attainment gap, for example by family income, differs "
        "between the ELS:2002 and HSLS:09 cohorts. The key assumption is that, "
        "without the change being studied, both groups would have followed "
        "parallel trends; the paper should report checks of it."
    ),
    "reliability": (
        "Reliability is how consistently a set of questionnaire items measures "
        "something, such as math self-efficacy: how much of the variation in "
        "scores is real differences between students rather than noise. It runs "
        "from 0 to 1; about 0.70 is often called acceptable for research and "
        "0.80 or more good. High reliability does not show the scale measures "
        "the RIGHT thing (that is validity)."
    ),
    "omega": (
        "Omega (McDonald's omega) is a reliability estimate for a scale, "
        "computed from a factor model of its items. Unlike Cronbach's alpha, it "
        "does not assume every item is an equally good indicator, so it is "
        "usually preferred. Read it like other reliability values (about 0.70 "
        "acceptable, 0.80 or more good); with only two or three items it is "
        "imprecise."
    ),
    "cfa": (
        "CFA (confirmatory factor analysis) tests whether questionnaire items "
        "hang together the way a theory says they should, for example that four "
        "items all reflect one 'math self-efficacy' factor. It produces loadings "
        "(how strongly each item reflects its factor) and fit indices such as "
        "CFI and RMSEA that say how well the model reproduces the data."
    ),
    "cfi": (
        "CFI (comparative fit index) compares a factor model with a baseline "
        "model in which items are unrelated. It runs from 0 to 1; about 0.95 or "
        "more is conventionally read as good fit and 0.90 as acceptable. These "
        "cut-offs are rules of thumb, not tests, and should be read together "
        "with RMSEA and the loadings."
    ),
    "rmsea": (
        "RMSEA (root mean square error of approximation) measures how badly a "
        "factor model misfits the data, per degree of freedom. Lower is better: "
        "0.06 or less is often read as good and above 0.10 as poor. With very "
        "few items (small degrees of freedom) RMSEA is unstable and can look bad "
        "for a reasonable model."
    ),
    "irt": (
        "IRT (item response theory) models how the chance of each answer to "
        "each item depends on a student's underlying level of the trait. The "
        "graded response model (GRM) is the IRT model for items with ordered "
        "options such as 'strongly disagree' to 'strongly agree'. It shows which "
        "items are most informative and at which trait levels the scale "
        "measures precisely."
    ),
    "dif": (
        "DIF (differential item functioning) means an item behaves differently "
        "for two groups of students who have the SAME level of the trait, for "
        "example boys and girls with equal math self-efficacy answering one "
        "item differently. It signals that the item may measure something extra "
        "for one group. Statistical DIF is common in large samples; judge it by "
        "its size, not only its p-value."
    ),
    "measurement invariance": (
        "Measurement invariance means a scale measures the same thing in the "
        "same way across groups, so group comparisons are fair. It is tested in "
        "steps: the same structure (configural), equal loadings (metric), and "
        "equal intercepts or thresholds (scalar). Means can only be compared "
        "between groups once scalar invariance holds, at least partly. The "
        "change in CFI (about 0.01 or less) is the usual guide."
    ),
    "cdm": (
        "A CDM (cognitive diagnosis model, such as DINA or G-DINA) classifies "
        "students as having mastered or not mastered each of several specific "
        "skills, using a table (the Q-matrix) that says which skills each item "
        "needs. Results depend heavily on that table being right. With few "
        "items per skill, classifications are uncertain, and the model can fail "
        "to separate skills at all; the paper should say so if it did."
    ),
    "lsar score": (
        "The LSAR score is a 1-10 rating from EDM-ARS's optional automated "
        "reviewer, which reads the finished paper the way a conference or "
        "journal reviewer might. Where the chosen venue has a benchmark (derived "
        "from papers it accepted), the score is compared with it; otherwise it "
        "is shown as a score only. It is a rough, noisy signal: two "
        "reviews of the same paper can differ by about two points. It is not "
        "peer review, not a prediction of acceptance and not a quality "
        "guarantee."
    ),
    "unverified": (
        "UNVERIFIED means EDM-ARS's internal methods reviewer (the Critic) "
        "still had unresolved concerns after the allowed number of revision "
        "rounds. The paper is written anyway, starts with a warning block and "
        "lists the concerns in an appendix. Read those concerns first: they are "
        "the most likely places the analysis is wrong."
    ),
    "incomplete": (
        "INCOMPLETE means the study ran to the end but the final checks found "
        "a problem serious enough that the paper is not released as finished, "
        "for example no PDF could be produced. The files are all kept, and "
        "`edmars results` explains what failed and what to do next."
    ),
    "invariant finding": (
        "An invariant finding comes from EDM-ARS's final automated checks, "
        "which compare the finished paper with the study's own result files "
        "without using AI: for example a number in the text that does not "
        "match the results, a table or figure that is referenced but missing, "
        "or a citation that is not in the bibliography. Each has a severity "
        "(critical, major, minor). Almost all are advisory: the paper is still "
        "produced, and each finding tells you exactly what to check by hand."
    ),
    "p-value": (
        "A p-value is the probability of seeing a result at least this extreme "
        "if there were really no effect or difference. Small values (below "
        "0.05 by convention) suggest the pattern is unlikely to be chance alone. "
        "With tens of thousands of students almost everything is 'significant', "
        "so look at the size of the effect and its confidence interval, not "
        "only the p-value."
    ),
    "test set": (
        "The test set is the 20% of students set aside before any model is "
        "trained or tuned. All reported prediction accuracy comes from these "
        "students, so it estimates how the model would do on new students from "
        "the same population, not how well it memorized the training data."
    ),
    "cross-validation": (
        "Cross-validation tunes and compares models using only the training "
        "data: it is split into several parts, and each part takes a turn as a "
        "mini test set. It chooses settings without ever looking at the real "
        "test set, which keeps the final accuracy estimate honest."
    ),
}

#: Other ways people write the same terms -> canonical key.
ALIASES: dict[str, str] = {
    "auc roc": "auc",
    "roc auc": "auc",
    "area under the curve": "auc",
    "roc": "auc",
    "root mean squared error": "rmse",
    "root mean square error": "rmse",
    "r2": "r-squared",
    "r squared": "r-squared",
    "r²": "r-squared",
    "rsquared": "r-squared",
    "ci": "confidence interval",
    "confidence intervals": "confidence interval",
    "95% ci": "confidence interval",
    "bootstrap": "confidence interval",
    "shapley": "shap",
    "shap values": "shap",
    "brier": "calibration",
    "brier score": "calibration",
    "fairness": "subgroup fairness",
    "subgroup": "subgroup fairness",
    "subgroups": "subgroup fairness",
    "average treatment effect": "ate",
    "att": "ate",
    "propensity": "propensity score",
    "propensity scores": "propensity score",
    "ps": "propensity score",
    "psm": "matching",
    "propensity score matching": "matching",
    "inverse probability weighting": "ipw",
    "iptw": "ipw",
    "inverse probability of treatment weighting": "ipw",
    "aipw": "doubly robust",
    "doubly-robust": "doubly robust",
    "dr": "doubly robust",
    "grf": "causal forest",
    "causal forests": "causal forest",
    "itr": "treatment rule",
    "individualized treatment rule": "treatment rule",
    "optimal treatment regime": "treatment rule",
    "treatment regime": "treatment rule",
    "did": "difference-in-differences",
    "diff-in-diff": "difference-in-differences",
    "difference in differences": "difference-in-differences",
    "gap-in-gaps": "difference-in-differences",
    "gap in gaps": "difference-in-differences",
    "mcdonald's omega": "omega",
    "mcdonalds omega": "omega",
    "ω": "omega",
    "alpha": "reliability",
    "cronbach's alpha": "reliability",
    "cronbachs alpha": "reliability",
    "confirmatory factor analysis": "cfa",
    "factor analysis": "cfa",
    "comparative fit index": "cfi",
    "tli": "cfi",
    "root mean square error of approximation": "rmsea",
    "srmr": "rmsea",
    "item response theory": "irt",
    "grm": "irt",
    "irt/grm": "irt",
    "graded response model": "irt",
    "differential item functioning": "dif",
    "invariance": "measurement invariance",
    "configural": "measurement invariance",
    "metric invariance": "measurement invariance",
    "scalar invariance": "measurement invariance",
    "cognitive diagnosis model": "cdm",
    "cognitive diagnostic model": "cdm",
    "dina": "cdm",
    "g-dina": "cdm",
    "gdina": "cdm",
    "lsar": "lsar score",
    "review score": "lsar score",
    "reviewer score": "lsar score",
    "gate": "lsar score",
    "not verified": "unverified",
    "critic_unverified": "unverified",
    "invariant": "invariant finding",
    "invariants": "invariant finding",
    "invariant findings": "invariant finding",
    "finding": "invariant finding",
    "findings": "invariant finding",
    "p value": "p-value",
    "pvalue": "p-value",
    "significance": "p-value",
    "held-out": "test set",
    "holdout": "test set",
    "test data": "test set",
    "cv": "cross-validation",
    "cross validation": "cross-validation",
}

_STRIP = re.compile(r"[\s_]+")


def _normalize(term: str) -> str:
    cleaned = term.strip().strip("\"'`?.!:").casefold()
    return _STRIP.sub(" ", cleaned).strip()


def lookup(term: str) -> str | None:
    """The canonical key for ``term`` (any spelling in ALIASES), or None."""
    key = _normalize(term)
    if key in TERMS:
        return key
    if key in ALIASES:
        return ALIASES[key]
    squashed = key.replace(" ", "-")
    if squashed in TERMS:
        return squashed
    spaced = key.replace("-", " ")
    if spaced in TERMS:
        return spaced
    if spaced in ALIASES:
        return ALIASES[spaced]
    return None


def _display_name(key: str) -> str:
    upper = {"auc", "rmse", "ate", "ipw", "cfa", "cfi", "rmsea", "irt", "dif", "cdm", "shap"}
    if key in upper:
        return key.upper()
    if key == "lsar score":
        return "LSAR score"
    if key == "p-value":
        return "p-value"
    if key in ("unverified", "incomplete"):
        return key.upper()
    return key[0].upper() + key[1:]


def term_names() -> list[str]:
    """Display names of every term, in the order :data:`TERMS` lists them."""
    return [_display_name(key) for key in TERMS]


def explain(term: str) -> str:
    """A plain-language explanation of ``term``, or suggestions when unknown."""
    key = lookup(term)
    if key is not None:
        return f"{_display_name(key)}\n\n{TERMS[key]}"
    candidates = list(TERMS) + list(ALIASES)
    close = difflib.get_close_matches(_normalize(term), candidates, n=3, cutoff=0.6)
    suggestions = sorted({_display_name(lookup(c) or c) for c in close})
    lines = [f'No explanation for "{term.strip()}" yet.']
    if suggestions:
        lines.append("Did you mean: " + ", ".join(suggestions) + "?")
    lines.append("Terms I can explain: " + ", ".join(term_names()) + ".")
    return "\n".join(lines)
