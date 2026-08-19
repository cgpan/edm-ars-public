# Certified helper: IRT calibration via mirt (P4).
# Input JSON: {items: {col: [...integer categories, NA ok...]},
#              itemtype: "graded" (Likert GRM) | "2PL" (binary)}
# Output: item parameters (discrimination a, thresholds b*), test
# information at theta grid, empirical (marginal) reliability, item fit.

args <- commandArgs(trailingOnly = TRUE)
input <- jsonlite::fromJSON(args[[1]])
out_path <- args[[2]]

res <- tryCatch({
  suppressMessages(library(mirt))
  dat <- as.data.frame(lapply(input$items, function(v) as.integer(v)))
  itype <- if (is.null(input$itemtype)) "graded" else input$itemtype
  m <- mirt::mirt(dat, 1, itemtype = itype, verbose = FALSE,
                  technical = list(NCYCLES = 2000))
  co <- mirt::coef(m, IRTpars = TRUE, simplify = TRUE)$items
  params <- lapply(seq_len(nrow(co)), function(i) {
    row <- co[i, ]
    list(item = rownames(co)[i],
         a = unname(row[["a"]]),
         b = unname(as.numeric(row[grep("^b", names(row))])))
  })
  theta <- seq(-3, 3, by = 0.25)
  ti <- mirt::testinfo(m, matrix(theta))
  rel <- mirt::marginal_rxx(m)
  ifit <- tryCatch({
    f <- mirt::itemfit(m, fit_stats = "S_X2")
    lapply(seq_len(nrow(f)), function(i)
      list(item = as.character(f$item[i]),
           S_X2 = f$S_X2[i], df = f$df.S_X2[i], p = f$p.S_X2[i]))
  }, error = function(e) NULL)
  list(
    itemtype = itype,
    params = params,
    theta_grid = theta,
    test_information = as.numeric(ti),
    marginal_reliability = rel,
    item_fit = ifit,
    converged = mirt::extract.mirt(m, "converged"),
    n = nrow(dat)
  )
}, error = function(e) list(error = conditionMessage(e)))

jsonlite::write_json(res, out_path, auto_unbox = TRUE, digits = 8, null = "null")
