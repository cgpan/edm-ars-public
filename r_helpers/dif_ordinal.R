# Certified helper: ordinal logistic-regression DIF (P5, lordif-style).
# Input JSON: {items: {col: [...integer categories...]}, group: [...],
#              purify_iters: 1}
# Per item: cumulative-logit models (MASS::polr)
#   M0: item ~ rest-score
#   M1: item ~ rest-score + group          (uniform DIF)
#   M2: item ~ rest-score * group          (+ non-uniform DIF)
# LR tests M0 vs M2 (overall), M0 vs M1 (uniform), M1 vs M2 (non-uniform);
# effect size = McFadden pseudo-R2 change (M0 -> M2). Flag rule: p<.01
# AND deltaR2 >= .02. NOTE: Jodoin & Gierl's .035/.07 cutoffs assume
# Nagelkerke R2; McFadden runs smaller (calibration sim: true 0.9-logit
# DIF -> ~.033, null items <= .002), so .02/.05 are the McFadden-scaled
# moderate/large bands, held to hit-rate/false-positive standards by
# scripts/psychometric_gates.py::dif_gate.
# rest-score = total of OTHER items (avoids self-contamination).

args <- commandArgs(trailingOnly = TRUE)
input <- jsonlite::fromJSON(args[[1]])
out_path <- args[[2]]

res <- tryCatch({
  suppressMessages(library(MASS))
  dat <- as.data.frame(lapply(input$items, function(v) as.integer(v)))
  grp <- factor(as.character(input$group))
  keep <- !is.na(grp) & rowSums(is.na(dat)) == 0  # listwise for DIF models
  dat <- dat[keep, , drop = FALSE]
  grp <- droplevels(grp[keep])

  mcfadden <- function(m, m0) 1 - (logLik(m) / logLik(m0))
  items <- names(dat)
  out <- list()
  for (it in items) {
    y <- factor(dat[[it]], ordered = TRUE)
    if (nlevels(y) < 2) next
    rest <- rowSums(dat[, setdiff(items, it), drop = FALSE])
    df0 <- data.frame(y = y, rest = scale(rest)[, 1], g = grp)
    null_m <- MASS::polr(y ~ 1, data = df0, Hess = TRUE)
    m0 <- MASS::polr(y ~ rest, data = df0, Hess = TRUE)
    m1 <- MASS::polr(y ~ rest + g, data = df0, Hess = TRUE)
    m2 <- MASS::polr(y ~ rest * g, data = df0, Hess = TRUE)
    lr02 <- 2 * (logLik(m2) - logLik(m0))
    df02 <- attr(logLik(m2), "df") - attr(logLik(m0), "df")
    lr01 <- 2 * (logLik(m1) - logLik(m0))
    df01 <- attr(logLik(m1), "df") - attr(logLik(m0), "df")
    lr12 <- 2 * (logLik(m2) - logLik(m1))
    df12 <- attr(logLik(m2), "df") - attr(logLik(m1), "df")
    r2_0 <- as.numeric(mcfadden(m0, null_m))
    r2_2 <- as.numeric(mcfadden(m2, null_m))
    d_r2 <- r2_2 - r2_0
    p_overall <- pchisq(as.numeric(lr02), df02, lower.tail = FALSE)
    out[[it]] <- list(
      item = it,
      p_overall = p_overall,
      p_uniform = pchisq(as.numeric(lr01), df01, lower.tail = FALSE),
      p_nonuniform = pchisq(as.numeric(lr12), df12, lower.tail = FALSE),
      delta_pseudo_r2 = d_r2,
      flagged = (p_overall < 0.01) && (d_r2 >= 0.02),
      effect_size_label = if (d_r2 >= 0.05) "large" else if
        (d_r2 >= 0.02) "moderate" else "negligible"
    )
  }
  list(
    items = unname(out),
    n = nrow(dat),
    groups = as.list(table(grp)),
    flag_rule = "p_overall < .01 AND McFadden delta pseudo-R2 >= .02 (moderate, McFadden-scaled)"
  )
}, error = function(e) list(error = conditionMessage(e)))

jsonlite::write_json(res, out_path, auto_unbox = TRUE, digits = 8, null = "null")
