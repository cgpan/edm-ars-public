# Certified helper: single-group CFA via lavaan (P3).
# Input JSON: {items: {colname: [values...]}, model: "lavaan model string",
#              estimator: "MLR"|"ML" (default MLR), std_lv: true}
# Missing values: nulls in JSON -> NA -> FIML (missing = "fiml"; requires
# ML-family estimator). Items are never imputed.
# Output JSON: {fit: {cfi,tli,rmsea,srmr,chisq,df,pvalue,n}, loadings:
#              [{factor,item,est_std,se}], converged, warnings}

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
  list(
    fit = as.list(fm),
    fit_robust = if (is.null(fm_r)) NULL else as.list(fm_r),
    loadings = lo,
    n = lavaan::lavInspect(fit, "nobs"),
    converged = lavaan::lavInspect(fit, "converged"),
    warnings = character(0)
  )
}, error = function(e) list(error = conditionMessage(e)))

jsonlite::write_json(res, out_path, auto_unbox = TRUE, digits = 8, null = "null")
