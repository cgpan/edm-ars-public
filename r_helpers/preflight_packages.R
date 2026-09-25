# Pre-flight probe, not an analysis helper: reports which of the R packages
# the certified helpers load are installed and loadable, so a psychometrics
# run can stop before any paid stage instead of failing inside the Analyst.
# Run through src.r_bridge.run_r_script with the same `Rscript --vanilla`
# invocation the helpers use. Base R only: jsonlite is itself one of the
# packages being checked, so the input file is ignored and the output JSON
# is written by hand.
# Output: {installed: [...], missing: [...], r_version: "4.4.1"}

args <- commandArgs(trailingOnly = TRUE)
out_path <- args[[2]]

pkgs <- c("jsonlite", "lavaan", "mirt", "CDM", "MASS")
ok <- vapply(pkgs, function(p) {
  isTRUE(suppressWarnings(suppressMessages(requireNamespace(p, quietly = TRUE))))
}, logical(1))

quote_all <- function(x) paste0('"', x, '"')
json_array <- function(x) {
  if (length(x) == 0) return("[]")
  paste0("[", paste(quote_all(x), collapse = ", "), "]")
}

json <- paste0(
  '{"installed": ', json_array(pkgs[ok]),
  ', "missing": ', json_array(pkgs[!ok]),
  ', "r_version": ', quote_all(paste(R.version$major, R.version$minor, sep = ".")),
  '}'
)
writeLines(json, out_path)
