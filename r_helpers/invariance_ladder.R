# Certified helper: measurement invariance ladder via lavaan (P6).
# Input JSON: {items: {col: [...]}, group: [...], model: "lavaan string"}
# Fits configural -> metric (loadings equal) -> scalar (+ intercepts equal)
# with FIML. Decision rules (Cheung & Rensvold / Chen): a step HOLDS if
# delta CFI >= -0.01 AND delta RMSEA <= 0.015 vs the previous step.
# Output: per-step fit, deltas, holds flags, highest_level_held.

args <- commandArgs(trailingOnly = TRUE)
input <- jsonlite::fromJSON(args[[1]])
out_path <- args[[2]]

res <- tryCatch({
  suppressMessages(library(lavaan))
  dat <- as.data.frame(lapply(input$items, function(v) as.numeric(v)))
  dat$.group <- as.character(input$group)
  dat <- dat[!is.na(dat$.group), ]

  fit_step <- function(equal) {
    lavaan::cfa(
      model = input$model, data = dat, group = ".group",
      group.equal = equal, estimator = "ML", missing = "fiml",
      std.lv = FALSE
    )
  }
  steps <- list(
    configural = character(0),
    metric = c("loadings"),
    scalar = c("loadings", "intercepts")
  )
  out <- list()
  prev <- NULL
  highest <- "none"
  for (nm in names(steps)) {
    f <- fit_step(steps[[nm]])
    fm <- lavaan::fitMeasures(f, c("cfi", "rmsea", "srmr", "chisq", "df"))
    entry <- list(fit = as.list(fm),
                  converged = lavaan::lavInspect(f, "converged"))
    if (!is.null(prev)) {
      d_cfi <- fm[["cfi"]] - prev[["cfi"]]
      d_rmsea <- fm[["rmsea"]] - prev[["rmsea"]]
      entry$delta_cfi <- d_cfi
      entry$delta_rmsea <- d_rmsea
      entry$holds <- (d_cfi >= -0.01) && (d_rmsea <= 0.015)
    } else {
      entry$holds <- entry$converged &&
        fm[["cfi"]] >= 0.90  # configural must at least fit adequately
    }
    out[[nm]] <- entry
    if (isTRUE(entry$holds) &&
        (nm == "configural" || isTRUE(out[[highest]]$holds))) {
      highest <- nm
    } else if (!isTRUE(entry$holds)) {
      # ladder stops at first failure
      prev <- fm
      break
    }
    prev <- fm
  }
  list(
    steps = out,
    highest_level_held = highest,
    groups = as.list(table(dat$.group)),
    decision_rule = "holds iff dCFI >= -0.01 and dRMSEA <= 0.015 (Chen 2007)"
  )
}, error = function(e) list(error = conditionMessage(e)))

jsonlite::write_json(res, out_path, auto_unbox = TRUE, digits = 8, null = "null")
