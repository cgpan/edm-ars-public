# Certified helper: single-group CFA via lavaan (P3).
# Input JSON: {items: {colname: [values...]}, model: "lavaan model string",
#              estimator: "MLR"|"ML" (default MLR), std_lv: true}
# Missing values: nulls in JSON -> NA -> FIML (missing = "fiml"; requires
# ML-family estimator). Items are never imputed.
# Output JSON: {fit: {cfi,tli,rmsea,srmr,chisq,df,pvalue,n}, loadings:
#              [{factor,item,est_std,se}], factor_cor: [{f1,f2,est_std}],
#              residual_cor: [{i1,i2,est_std}], converged, warnings}
#
# factor_cor carries the STANDARDIZED latent covariances, diagonal
# included. psy_omega() needs them: omega-total over a multi-factor model
# is (1'L Phi L'1) / (1'L Phi L'1 + sum theta), and without Phi the only
# computable quantity is a per-factor omega. The version of this file
# that omitted Phi is why one delivered paper reported omega-total 0.922
# for a two-factor scale whose admissible range was [0.885, 0.904]:
# 0.922 is what the UNIDIMENSIONAL formula returns, i.e. Phi taken to be
# all ones. residual_cor exists so psy_omega can refuse, rather than
# under-report, when the model declares correlated residuals.

args <- commandArgs(trailingOnly = TRUE)
input <- jsonlite::fromJSON(args[[1]])
out_path <- args[[2]]

res <- tryCatch({
  suppressMessages(library(lavaan))
  dat <- as.data.frame(lapply(input$items, function(v) as.numeric(v)))
  est <- if (is.null(input$estimator)) "MLR" else input$estimator
  fit <- lavaan::cfa(
    model = input$model,
    data = dat,
    estimator = est,
    missing = "fiml",
    std.lv = if (is.null(input$std_lv)) TRUE else isTRUE(input$std_lv)
  )
  fm <- lavaan::fitMeasures(
    fit, c("cfi", "tli", "rmsea", "srmr", "chisq", "df", "pvalue")
  )
  # robust variants when MLR
  fm_r <- tryCatch(
    lavaan::fitMeasures(fit, c("cfi.robust", "tli.robust", "rmsea.robust")),
    error = function(e) NULL
  )
  std <- lavaan::standardizedSolution(fit)
  lo <- std[std$op == "=~", c("lhs", "rhs", "est.std", "se")]
  names(lo) <- c("factor", "item", "est_std", "se")

  # Standardized latent covariances (Phi) and residual covariances
  # (off-diagonal Theta). `~~` rows cover both; split them by whether the
  # name is a declared factor.
  factors <- unique(as.character(lo$factor))
  cov_rows <- std[std$op == "~~", c("lhs", "rhs", "est.std")]
  is_lat <- cov_rows$lhs %in% factors & cov_rows$rhs %in% factors
  fc <- cov_rows[is_lat, ]
  names(fc) <- c("f1", "f2", "est_std")
  rc <- cov_rows[!is_lat & cov_rows$lhs != cov_rows$rhs, ]
  names(rc) <- c("i1", "i2", "est_std")

  list(
    fit = as.list(fm),
    fit_robust = if (is.null(fm_r)) NULL else as.list(fm_r),
    loadings = lo,
    factor_cor = fc,
    residual_cor = rc,
    n = lavaan::lavInspect(fit, "nobs"),
    converged = lavaan::lavInspect(fit, "converged"),
    warnings = character(0)
  )
}, error = function(e) list(error = conditionMessage(e)))

jsonlite::write_json(res, out_path, auto_unbox = TRUE, digits = 8, null = "null")
