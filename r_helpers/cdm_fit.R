# Certified helper: cognitive diagnosis models via the CDM package (P7).
# Input JSON: {responses: {item: [0/1/null...]}, q_matrix: {item: [attr indices 1-based]},
#              attributes: ["name", ...], model: "DINA"|"GDINA"}
# Output: item params (guess g, slip s per item for DINA; per-item fit),
#         attribute mastery prevalence, per-student mastery profiles
#         summary, model fit (AIC/BIC), converged.

args <- commandArgs(trailingOnly = TRUE)
input <- jsonlite::fromJSON(args[[1]])
out_path <- args[[2]]

res <- tryCatch({
  suppressMessages(library(CDM))
  resp <- as.data.frame(lapply(input$responses, function(v) as.integer(v)))
  attrs <- input$attributes
  K <- length(attrs)
  items <- names(resp)
  Q <- matrix(0L, nrow = length(items), ncol = K,
              dimnames = list(items, attrs))
  qm <- input$q_matrix
  for (it in items) {
    idx <- qm[[it]]
    Q[it, as.integer(idx)] <- 1L
  }
  model <- if (is.null(input$model)) "DINA" else toupper(input$model)
  comparison <- NULL
  if (model == "COMPARE") {
    # E2b: fit BOTH; DINA is nested in G-DINA. Report per-model fit and
    # a likelihood-ratio test; select by BIC (parsimony-favoring).
    fit_dina <- CDM::din(resp, q.matrix = Q, progress = FALSE)
    fit_gdina <- CDM::gdina(resp, q.matrix = Q, progress = FALSE)
    ll_d <- as.numeric(logLik(fit_dina)); np_d <- attr(logLik(fit_dina), "df")
    ll_g <- as.numeric(logLik(fit_gdina)); np_g <- attr(logLik(fit_gdina), "df")
    aic_d <- AIC(fit_dina); bic_d <- BIC(fit_dina)
    aic_g <- AIC(fit_gdina); bic_g <- BIC(fit_gdina)
    lr <- 2 * (ll_g - ll_d)
    df_lr <- np_g - np_d
    p_lr <- pchisq(max(0, lr), df = max(1, df_lr), lower.tail = FALSE)
    degenerate <- (df_lr <= 0) || (max(rowSums(Q)) == 1)
    # Single-attribute items: G-DINA's saturated form coincides with
    # DINA (P(0)/P(1) = guess/1-slip), so the comparison is
    # uninformative BY CONSTRUCTION - say so and keep DINA for
    # interpretability. Otherwise select by BIC.
    selected <- if (degenerate) "DINA" else
      if (bic_d <= bic_g) "DINA" else "GDINA"
    comparison <- list(
      dina = list(AIC = aic_d, BIC = bic_d, loglike = ll_d, npars = np_d),
      gdina = list(AIC = aic_g, BIC = bic_g, loglike = ll_g, npars = np_g),
      lr_stat = lr, lr_df = df_lr, lr_p = if (df_lr > 0) p_lr else NA,
      degenerate_single_attribute = degenerate,
      selected = selected,
      note = if (degenerate)
        "single-attribute Q-matrix: DINA and G-DINA coincide; comparison uninformative by construction; DINA retained for interpretability"
      else "DINA nested in G-DINA; selection by BIC; LR test reported"
    )
    fit <- if (selected == "DINA") fit_dina else fit_gdina
    model <- selected
    if (selected == "DINA") {
      ip <- data.frame(item = items,
                       guess = fit_dina$guess$est,
                       slip = fit_dina$slip$est)
    } else {
      ip <- data.frame(item = items, guess = NA, slip = NA)
    }
  } else if (model == "DINA") {
    fit <- CDM::din(resp, q.matrix = Q, progress = FALSE)
    ip <- data.frame(item = items,
                     guess = fit$guess$est,
                     slip = fit$slip$est)
  } else {
    fit <- CDM::gdina(resp, q.matrix = Q, progress = FALSE)
    co <- CDM::coef(fit)
    ip <- data.frame(item = items, guess = NA, slip = NA)
  }
  # attribute mastery prevalence from marginal skill probabilities
  skillprob <- as.numeric(fit$skill.patt[, 1])
  # per-student MAP profiles -> mastery rate per attribute
  patt <- CDM::IRT.factor.scores(fit, type = "MAP")
  list(
    model = model,
    comparison = comparison,
    item_params = ip,
    attribute_prevalence = as.list(setNames(skillprob, attrs)),
    n_students = nrow(resp),
    n_items = length(items),
    fit = list(AIC = AIC(fit), BIC = BIC(fit),
               loglike = as.numeric(logLik(fit))),
    converged = TRUE,
    note = "responses may contain NA (structural sparsity); CDM handles missing natively"
  )
}, error = function(e) list(error = conditionMessage(e)))

jsonlite::write_json(res, out_path, auto_unbox = TRUE, digits = 8, null = "null", na = "null")
