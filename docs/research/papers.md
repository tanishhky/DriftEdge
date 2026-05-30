# Research Foundation — Annotated Bibliography

The literature underpinning DriftEdge. Each entry has a one-paragraph annotation explaining *why it matters for path-favorable prediction-market trades*.

---

## Part 1 — Foundations of prediction markets

### Wolfers, J., & Zitzewitz, E. (2004)
**"Prediction Markets."** *Journal of Economic Perspectives*, 18(2), 107–126. NBER WP w10504.

The canonical survey. Describes contract types, applications, design issues, and the conditions under which prediction markets aggregate information efficiently. Documents that markets *typically* outperform polls and expert panels, but are not perfectly calibrated, especially on small-probability events. *Why for us:* Sets the prior that there is real signal in prices, but also that there are systematic deviations from efficiency (small-probability miscalibration, longshot bias) — both of which create tradeable structure.

### Hanson, R. (2003)
**"Combinatorial Information Market Design."** *Information Systems Frontiers*, 5(1), 107–119.

Introduces the Logarithmic Market Scoring Rule (LMSR), the dominant prediction-market mechanism for years. *Why for us:* Polymarket has migrated to a CLOB since 2024, but many academic results assume LMSR. Understanding the mechanism difference (LMSR has automatic market-making liquidity; CLOB has discrete order books) is essential to translate older results to today's market structure.

### Berg, J. E., Forsythe, R., Nelson, F., & Rietz, T. (2008)
**"Results from a Dozen Years of Election Futures Markets Research."** *Handbook of Experimental Economic Results*.

Empirical performance of the Iowa Electronic Markets across 12 years of US elections. *Why for us:* Real evidence that markets beat polls, and a baseline for how much "real-world information" leaks into prices ahead of news events. The structure we're trying to detect (information showing up in flow before showing up in marginal price) is what makes this whole thesis work.

---

## Part 2 — Path-dependence and microstructure

### Anonymous (2025)
**"Path Dependence in AMM-Based Markets: Mathematical Proof and Implications for Truth Discovery."** arXiv 2503.00201.

Proves that AMM-based prediction markets are inherently path-dependent: the *sequence* of trades matters for the final state, not just the aggregate. *Why for us:* Even though Polymarket has moved to CLOB, the order in which trades arrive still imprints on the book through limit-order placement. Path-dependence at the microstructure level is exactly what the flow engine exploits.

### Anonymous (2025)
**"How manipulable are prediction markets?"** arXiv 2503.03312.

A randomized field experiment shocking prices on real prediction markets. Finds that price shocks remain visible **60 days later** — informed (or even uninformed) flow does not get arbitraged away quickly. *Why for us:* This is the single strongest empirical justification for our trade. If shocks persist for 60 days, our 0.36 → 0.60 path target is a slow-moving phenomenon we have time to detect, position, and exit on.

### Anonymous (n.d.)
**"The Informational Content of the Limit Order Book: An Empirical Study of Prediction Markets."** arXiv 1609.03471.

Empirical study of how the limit order book (vs just the top-of-book mid) contains predictive information in prediction markets. *Why for us:* The flow engine should consume the *full* book (bid/ask sizes at multiple levels), not just the trade tape. Imbalance metrics from the book are likely to be the strongest single feature.

### Anonymous (2025)
**"SoK: Market Microstructure for Decentralized Prediction Markets (DePMs)."** arXiv 2510.15612.

A systematization-of-knowledge paper on DePM microstructure — covers Polymarket specifically alongside Augur, Gnosis, etc. *Why for us:* Practitioner-grade reference for how Polymarket's CLOB differs from traditional equity CLOBs, and where adversarial dynamics (MEV, oracle manipulation) introduce noise we need to filter.

---

## Part 3 — Inference and information extraction

### Anonymous (2026)
**"Prediction Markets as Bayesian Inverse Problems: Uncertainty Quantification, Identifiability, and Information Gain from Price-Volume Histories under Latent Types."** arXiv 2601.18815.

Formalizes prediction markets as Bayesian inverse problems: given an observed history of `(p_t, V_t)`, infer the latent distribution of trader types and the future outcome. *Why for us:* This is the conceptual frame for the path engine. We are not trying to "predict the event" — we are trying to identify the regime (informed flow vs noise vs manipulation) currently driving the price.

### Aït-Sahalia, Y., & Lo, A. W. (1998)
**"Nonparametric Estimation of State-Price Densities Implicit in Financial Asset Prices."** *Journal of Finance*, 53(2), 499–547.

Kernel-based density estimation from option prices. *Why for us:* Conceptually, what we want from a sequence of prediction-market prices is a non-parametric estimate of the *probability path's drift and volatility*. The same machinery (kernel smoothing, bandwidth selection) translates.

---

## Part 4 — Sizing: Kelly criterion

### Kelly, J. L. (1956)
**"A New Interpretation of Information Rate."** *Bell System Technical Journal*, 35(4), 917–926.

The original Kelly paper. *Why for us:* Foundational. The whole sizing engine is built on this — but with the prediction-market-specific form `f* = (p - c) / (1 - c)`, not the sports-betting `(bp - q) / b` form.

### MacLean, Thorp, & Ziemba (2011, eds.)
**"The Kelly Capital Growth Investment Criterion: Theory and Practice."** World Scientific.

The standard reference on fractional Kelly, drawdown control, and practical Kelly application. Documents the 33% probability-of-halving-bankroll for full Kelly. *Why for us:* Justifies why DriftEdge defaults to quarter-Kelly (`0.25 · f*`), capturing ~75% of long-term growth with ~25% of the variance.

### Anonymous (2024)
**"Application of the Kelly Criterion to Prediction Markets."** arXiv 2412.14144.

Modern paper on Kelly sizing specifically for prediction-market structure, including how to handle estimation error in `p` (using shrinkage). *Why for us:* The exact formulation we'll implement. Includes practical handling of the case where our `p_estimated` is itself uncertain — relevant since the path engine's output is necessarily noisy.

---

## Part 5 — Favorite-longshot bias and behavioral structure

### Levitt, S. D. (2004)
**"Why Are Gambling Markets Organised So Differently from Financial Markets?"** *Economic Journal*, 114, 223–246.

Foundational paper documenting bookmaker behavior in sports betting. Shows that books don't run pure balanced order books; they exploit known bettor biases. *Why for us:* Same biases (favorite-longshot, narrative-driven flow) appear in prediction markets. Useful for understanding which markets are most likely to be mispriced.

### Whelan, K. (2024)
**"Risk Aversion and Favourite-Longshot Bias in a Competitive Fixed-Odds Betting Market."** *Economica*.

Modern theoretical treatment of favorite-longshot bias under competitive book-making. *Why for us:* Establishes when the bias should and shouldn't appear. Prediction markets without a designated market-maker show varying patterns; this helps us reason about which markets are likely to be calibrated.

### Snowberg, E., & Wolfers, J. (2010)
**"Explaining the Favorite-Long Shot Bias: Is it Risk-Love or Misperceptions?"** *Journal of Political Economy*, 118(4), 723–746.

Decomposes the bias into rational risk-loving vs cognitive misperception. Concludes it's mostly misperception. *Why for us:* If the bias is misperception-driven, it's stable and tradeable; if it's risk-preference-driven, it's hedonic and we shouldn't expect it to mean-revert. The misperception finding is good news for systematic strategies.

---

## Reading order for new contributors

1. Wolfers & Zitzewitz 2004 (just the survey, skim Section 5)
2. Kelly 1956 (you can read it in 20 minutes)
3. arXiv 2503.03312 (How Manipulable Are Prediction Markets — the empirical anchor)
4. arXiv 2412.14144 (Application of the Kelly Criterion to Prediction Markets — the operational reference)
5. arXiv 1609.03471 (Limit Order Book information content — for the flow engine)
6. arXiv 2601.18815 (Bayesian inverse formulation — conceptual frame for the path engine)

Everything else as you go deeper into specific subsystems.
