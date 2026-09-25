# Disclaimer

EDM-ARS is an experimental research-assistance tool. Please read this before
you use it or anything it produces. By installing or running EDM-ARS you
confirm that you have read and accept this disclaimer and the
[privacy notice](PRIVACY.md).

## What EDM-ARS produces

EDM-ARS produces **AI-generated drafts**: research questions, analysis code,
statistical results, figures, citations and complete manuscripts. They are
produced by large language models and automated checks, not by a human expert.

- **Drafts can be wrong.** They can contain incorrect numbers, misread
  variables, analysis mistakes, citations that are invented, misattributed or
  do not say what the paper claims, overstated conclusions, and text that
  resembles existing work. The automated checks catch many of these problems,
  not all of them.
- **You are responsible for everything you use.** Before you share, submit,
  publish, teach with or act on any output, you must check every number,
  every citation and every claim yourself, and confirm the analysis is
  appropriate for your question.
- **Authorship and AI disclosure are your responsibility.** Follow the rules of
  your journal, conference, funder and institution on AI assistance, and keep
  the automated-generation disclosure that EDM-ARS writes into each paper.
- **Reviewer scores are rough signals.** The optional automated reviewer
  (LSAR) gives noisy scores (two readings of the same paper can differ by about
  two points on a ten-point scale). A score is not peer review, not a
  prediction of acceptance, and not a quality guarantee.
- **Not for decisions about individual students.** Prediction models trained by
  EDM-ARS describe patterns in historical survey data and can be inaccurate or
  unfair for some groups. Do not use them to make or support high-stakes
  decisions (placement, discipline, admission, funding, services) about real
  individuals.

## Data you use

- Use EDM-ARS **only with public-use data** whose terms allow this kind of
  analysis. Never use restricted-use or licensed data, data covered by a
  data-use agreement that forbids sending it to third-party services, or any
  data about identifiable people.
- You are responsible for following the terms of each dataset (for example the
  NCES public-use terms: no attempt to identify individuals, and cite the
  source) and for any ethics or IRB requirements that apply to your work.
- EDM-ARS is not affiliated with, sponsored by or endorsed by NCES, the
  Institute of Education Sciences, or any other data provider.

## Code that runs on your computer

EDM-ARS asks an AI model to write Python (and sometimes R) code, and then
**runs that code on your computer, with your user permissions, without an
isolated sandbox**. The code is meant to read the dataset and write results
into the study folder, and EDM-ARS removes your API keys from its
environment, but AI-written code can contain mistakes. It can use a lot of
memory, processor time and disk space, and in rare cases it could modify or
delete files your user account can reach. Keep backups of important files and
do not run EDM-ARS under an administrator account.

## Third-party services and costs

- EDM-ARS sends requests to third-party services that **you** choose and pay
  for, such as DeepSeek, OpenAI or Anthropic, and to free services such as
  Semantic Scholar, arXiv and Crossref. Their own terms, prices and privacy
  policies apply. Some providers process data outside your country; check
  whether your institution allows the provider you pick. See
  [PRIVACY.md](PRIVACY.md) for exactly what is sent.
- **You are responsible for all charges** on your accounts. Cost figures
  EDM-ARS shows are estimates based on a small number of measured runs and
  published prices, and can be wrong. Set a spending limit with your provider.
- EDM-ARS is not affiliated with, sponsored by or endorsed by DeepSeek,
  OpenAI, Anthropic, Semantic Scholar, arXiv, Crossref or any other service
  it connects to. Services and datasets can change or disappear at any time.

## No warranty, no liability

EDM-ARS and LSAR are provided free of charge under the MIT License, **"as is",
without warranty of any kind**, express or implied, including fitness for a
particular purpose, accuracy, or non-infringement. To the fullest extent
permitted by law, the authors and contributors are not liable for any claim,
damages, costs or other liability arising from the use of EDM-ARS or its
outputs, including academic, professional, financial or data-related harm.
See [LICENSE](LICENSE) for the full terms.

This disclaimer is not legal advice. If you are unsure whether a use is
allowed, ask your institution.
