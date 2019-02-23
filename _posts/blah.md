## Counterfactual evaluation of recommender systems

Most systems that we have on the internet are some sort of user-interactive systems where the user gives the system some context/intent, and the system produces a ranked list of items. Examples of such systems include search engines, social media feeds, news feeds etc. A classic way of evaluating how good a system performs offline would be something akin to mean-average-precision-at-K (MAP@K) or normalized-discounted-cumulative-gain (NDCG)  where the score is high when the items a user interacted with are at the top of the list. This is quick and easy if you have holdout some user feedback to validate, and you can get unbiased estimates of MAP@K and NDCG.

However, as most people in industry will attest, no one _actually_ cares about NDCG or MAP because it is often seen that optimizing such metrics doesn't necessarily increase KPIs such as _ad-revenue_ or _click-through-rate_. It is therefore vital to run online A/B tests to verify whether improving such offline metrics actually makes a different to the production system. There are a few reasons why discrepancies exist, and one of the biggest ones is the huge _presentation and sampling bias_ that exists in the user feedback logs, on which the offline metrics are evaluated. Given the nature of the systems, users only provide feedback on items that are shown to the user which makes it problematic to evaluate newer systems that may recommend items that were never shown to the user.

To counter this, we can always have a small portion of a userbase that are always given uniformly random results and the feedback we observe on these results would not have the same sampling bias of the existing algorithm in production. This can obviously be an extremely hard sell to any product people to show a portion of users completely bogus results but if this is done, we can mitigate a lot of the afforementioned sampling bias from validation data. Since showing uniformly random results is mostly impractical\*, this post explores alternative possibilities. Furthermore, it would be great if we could directly evaluate/estimate the KPIs offline instead of always having to rely on running expensive A/B tests that can be quite expensive as some tests need to run for a significant amount of time to feel confident in the results.

### User-interactive systems and partial information setting

In user-logs, we can collect data of the form $$ \mathcal{D} = [(x_1, y_1, r_1), \dots , (x_n, y_n, r_n)] $$ where $$ \boldsymbol x $$ is a user context, $$ \boldsymbol y $$ is the output of the system in production and $$ \boldsymbol r $$ is the reward/feedback (e.g. click, time-spent, ad-click etc.) observed from the user on the recommendation, and an ideal system would produce $$ \boldsymbol y $$ such that the reward $$ r $$ is maximized.


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

