---
layout: post
title:  "Simulating A/B tests offline with counterfactual inference"
date:   2018-12-10 15:07:19
categories: [intro, counterfactuals]
comments: false
---

While developing machine learning (ML) algorithms in production environments, we usually optimize a function or a loss that has nothing to do with our business goals. For example, we might actually care about metrics such as _click-through-rate_ or _average time-spent_ or _ad-revenue_ but our machine learning algorithms often minimize log-loss or root mean squared error (RMSE) of arbitrary quantities for computational ease. It is often seen that these machine learning metrics don't correlate with the original targets, which is why running online A/B tests is so vital in production because it lets us verify our ML models against the true objectives of a task. 

However, running A/B tests can be quite expensive because you need to launch some productionized version of an experiment, and let it run for a significant amount of time to get reliable results, during which you also risk potentially exposing a poorer system to users. This is why reliable offline evaluation is absolutely critical because it can allow a business to experiment more in a sandbox environment as well as provide intuitions on what is worth launching/testing in production.

<!--more-->

### Classic offline evaluation of user-interactive systems

A user-interactive system (e.g. product search, social media feeds, news recommendation etc.) involves showing a user a set of items given a certain user context or intent.  In such systems, users can only click/like/dislike the items recommended by the system. A classic way to evaluate how good such a system performs offline would be something akin to [mean average precision (MAP)](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision) or [normalized discounted cumulative gain (NDCG)](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Discounted_cumulative_gain) where you store historical user interactions, and verify your system by checking whether the algorithm places the items a user interacted with at the top of the recommendations.

While it has been common practice to evaluate systems using these kinds of approaches but it is also seen that optimizing such metrics don't necessarily increase the business KPIs. There are a few reasons why this discrepancy exists, and one of the biggest ones is the huge _presentation and sampling bias_ that exists in the user feedback logs on which offline evaluation metrics are calculated. Given the nature of the systems, it is problematic to evaluate newer algorithms that may have recommended items that were never shown to the user in the first place and evaluating on aggregate data can result in _confounding_ results. 

For example, imagine in a news recommendation problem, you were trying to evaluate two algorithms that performed differently for two different kinds of content (major publishers and long-tail). On first glance, it may be obvious to say that _Algorithm 2_ is probably "better" than _Algorithm 1_ but in industry it is often the case that you can have _Algorithm 1_ running in production, and you may have realized that it performs poorly for long-tail content and hence mostly only recommended articles from the major publishers, and wildly overestimate the true performance of an algorithm. 

<!-- <div id="simpsonsParadox"></div>

<div style="text-align:center"><div style="display:inline-block;">
<button type="button" class="btn btn-default" id="randomize_usage">
  <span class="glyphicon glyphicon-refresh"></span> Randomize
</button>
</div></div>

<script src="https://d3js.org/d3.v4.min.js"></script> -->

<!-- <script src="/assets/alias_table.js"></script>
<script>
var table = new AliasTable([0.4, 0.3, 0.2, 0.08, 0.02]);
table.displayState(d3.select("#aliastable"), 0);
function randomizeAliasTable() {
    table.stop = true;
    table = new AliasTable([Math.random(), Math.random(), Math.random(), Math.random(), Math.random()]);
    table.displayState(d3.select("#aliastable"), 0);
}
d3.select("#randomize_aliastable").on("click", randomizeAliasTable);
</script> -->
<!-- 
<script src="/assets/simpsons_paradox.js"></script>

<script>
var table = new Usage([0.8, 0.3]);
table.displayState(d3.select("#simpsonsParadox"), 0);
function randomize() {
   table.stop = true;
   table = new Usage([Math.random(), Math.random()]);
   table.displayState(d3.select("#simpsonsParadox"), 0);
}
d3.select("#randomize_usage").on("click", randomize);
</script> -->

<!-- | | Cohort 1 | Cohort 2 |
|:--------|:-------:|--------:|
| Article 1   | 0.7   | 0.2   |
| Article 2   | 0.8   | 0.3   |
|---- -->

Had the users been exposed to the true distribution of major publishers and long tail content by our algorithm in production, we could fairly evaluate our algorithms offline, and we can always pick a small fraction of users that are always given uniformly random results and the feedback we observe on these results would not have the same sampling bias of the existing algorithm in production. This can obviously be an extremely hard sell to any business/product to show some users completely bogus results but if this is done, we can mitigate a lot of the afforementioned sampling bias from our validation data. Since showing uniformly random results is mostly impractical\*, this post explores practical alternative possibilities for offline evaluation using counterfactual thinking.

### Offline evaluation as a regression problem

In user-interactive systems, it is generally easy to collect a dataset $$ \mathcal{D} = [(x_1, y_1, r_1), \dots, (x_n, y_n, r_n)] $$ where the $$ x_i $$ is what a system knows about the user and her intent, $$ y_i $$ is the _item_ recommended to user by some "algorithm" $$ \mathcal{A_0} $$, and $$ r_i $$ is the _reward_ (like/view/dislike) that a user provides to the system. This reward can be both implicit (view) or explicit (upvote). Many recommendation algorithms such as [collaborative filtering](http://yifanhu.net/PUB/cf.pdf) use this dataset to learn algorithms that can generate better $$ \textbf{y} $$. In contrast, an offline estimator's goal is to estimate the rewards $$ \textbf{r} $$ given a new set of $$ \textbf{y} $$ generated by a different algorithm $$ \mathcal{A} $$.

Given this dataset, it is relatively straightforward to simply learn a reward predictor $$ \hat{\textbf{r}} $$ as a function of the user-context and the recommended items if we think of the offline estimation problem as a classic regression problem. 

$$
min_{\textbf{w}} = \frac{1}{2} \sum_{i=1}^n (\hat{\textbf{r}}(x_i, y_i, \textbf{w}) - r_i)^2 + \frac{\lambda}{2} \left\lVert w\right\rVert^2
$$

Unfortunately, this doesn't work very well because the sampling bias in our data plays a huge role, and even with heavy regularization, the reward predictor is generally found to be not very indicative of true evaluations. 

Modeling the reward as a function of these two inputs can be problematic as it introduces modeling bias because often a user feedback can have external reasons outside of what the dataset captures. For example, a user may choose to ignore a longread news recommendation while queueing for coffee at a cafe but may have loved the recommendation in other uncaptured contexts.

### Counterfactual Thinking

Another reason why a vanilla reward predictor doesn't perform that well is because offline evaluation is fundamentally an interventional problem as opposed to observational. We only _observe_ the logged data generated by an algorithm $$ A_0 $$ in production, and we try to estimate _what if_ the users were recommended some other items. There are some intricate differences in observing $$ \textbf{y} $$ in logs and _setting_ $$ \textbf{y} $$ to different values (counterfactuals) and trying to estimate the reward. [Ferenc Huszár](https://twitter.com/fhuszar) has some really good [blog-posts](https://www.inference.vc/untitled/) on the intricate differences on this, and I highly recommend them.

### Inverse Propensity Scoring

Given a counterfactual framework, modeling the bias in the logs directly as opposed to modeling the reward works much better for offline evaluation. One classic way to do that would be [Inverse Propensity Scoring (IPS)](http://www.rebeccabarter.com/blog/2017-07-05-ip-weighting/) which evaluates treatments independent of the logged confounders. The key idea in IPS is to reweight samples based on _propensities_ of the logged items. Propensity $$ p_{x_i, y_i} $$ in this context is simply the probability of the item $$ y_i $$ shown to the user $$ x_i $$ at the point of logging. In user-interactive systems, it is generally an _algorithm_ that decides which item to show to a user, and it is generally straightforward to also log the probability o






<!-- 

### User-interactive systems and the partial information setting

A user-interactive system (e.g. product search, social media feeds etc.) involves showing a user a set of items $$ y_i $$ given a certain user context or intent $$ x_i $$.  In such systems, users can only provide feedback $$ r_i $$ such as clicks/likes/dislikes/ratings etc. Many ML models aggregate this data, and learn a model but this can be problematic because the averaged observations can be very confounding.

For a toy example, ...
You see [Simpson's paradox]() like this all the time in production because of the huge _presentation and sampling bias_ that exists in the data. 

   If all this data was collected in logs, we could have a dataset $$ \mathcal{D} = [(x_1, y_1, r_1), \dots , (x_n, y_n, r_n)] $$, and an ideal system would produce $$ \boldsymbol y $$ such that the reward $$ \boldsymbol r $$ is maximized. This formulation is identical to a [contextual bandit problem](https://arxiv.org/abs/1003.0146) and the only assumption made is that $$ \boldsymbol r $$ is dependent on $$ \boldsymbol x $$ and $$ \boldsymbol y $$. This reward can be anything -- like a click or time-spent or even something like ad-revenue.

Considering that it is relatively easy to collect a lot of such logs, a simplistic thing to try could be to learn $$ \hat{\boldsymbol r}(\boldsymbol x, \boldsymbol y) $$ as a function of the context and items from our log data $$ \mathcal{D} $$. This turns out to be problematic because of the huge _presentation and sampling bias_ that exists in the dataset as we only observe feedback on items a production algorithm showed to the users. Modeling $$ \hat{\boldsymbol r} $$ as a function of $$ x $$ and $$ y $$ also introduces _modeling bias_ because a reward can have dependency on many factors not captured by our context and feeds. 

This is not just a problem that is prevalent in A/B test simulations -- even extremely popular recommendation algorithms such as [collaborative filtering with implicit feedback](http://yifanhu.net/PUB/cf.pdf) interpret non-clicks as uninteresting items for the user but this can be problematic if the system never explored the possibility of showing certain categories of such items to the users, and the feedback we receive from users is [missing-not-at-random (MNAR)](http://www.cs.cornell.edu/~ylongqi/publication/recsys18a/). 

### Counterfactual Thinking and Importance Sampling

some user context $$ x_u $$, some feed $$ y_u $$ that you show to the user (news-feed/search or product results), and you observe *some* reward $$ \delta $$ (like a click, or "revenue" made during the session etc.) on the recommended items. Traditionally, you'd generally create a usage matrix $$ U $$ where each cell would represent some observed usage



Inverse Propensity Scoring

Three Ingredients:
1. Stochasticity and propensity

Most user-interactive systems that are in _exploitation_ mode, are _almost_ deterministic. Many recommender systems score a list of items based on content or collaborative (or both) filtering, and then rank/sort before showing them to the user. This can be problematic because this _greedy_ approach  -->


