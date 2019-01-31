---
layout: post
title:  "Counterfactual Evaluation"
date:   2018-12-06 15:07:19
categories: [intro, counterfactuals]
comments: false
---

When I was at [RecSys](https://recsys.acm.org/recsys18/) earlier this year, I learned about using [Inverse Propensity Scoring](https://en.wikipedia.org/wiki/Inverse_probability_weighting) to evaluate recommender systems using just offline data. When Susan Athey talked about counterfactual inference at [NeurIPS](https://neurips.cc/Conferences/2018/Schedule?showEvent=10982), I got inspired to write about my learnings with playing around with this technique for the past few weeks. Since most deployed production systems tend to be user-interactive, this is applicable in more than just a recommender system context.

<!--more-->

While developing machine learning systems, we often optimise losses (such as RMSE or log-loss) that don't necessarily correlate with engagement KPIs. That is why running A/B tests is vital to verify our algorithms. However, A/B tests have their own flaws -- to be confident in the results, the tests need to be run for a significant period of time (often a couple of weeks) and there is always a risk of exposing poor systems to significant portions of a userbase. Wouldn't it be great if we could estimate a system's true performance purely using offline data? In a counterfactual framework, recent research shows that it is possible to simulate A/B tests offline for user-interactive systems.

### User-interactive systems and counterfactual thinking

User-interactive systems involve showing users a set of items given a certain context (product search, or social media feeds). Such a system is given some user context and intent $$ x_u $$ and is required to produce a list of items $$ y_u $$ as a result. In such environments, systems only observe feedback on items that are shown to the user. 



However, while developing machine learning algorithms, we often optimise losses such as root-mean-squared error (RMSE) or log-loss and often find that reducing such a metric doesn't necessarily correlate with the KPIs that we *actually* care about.

In applications where the number of items is *much* larger than the feed length, it is sometimes non-optimal to optimize for clicks when we don't observe user interaction on majority of the items.

[Collaborative filtering with implicit feedback](http://yifanhu.net/PUB/cf.pdf) interprets non-clicks as uninteresting items for the user but this can be problematic if the system never explored the possibility of showing certain categories of such items to the users. 

some user context $$ x_u $$, some feed $$ y_u $$ that you show to the user (news-feed/search or product results), and you observe *some* reward $$ \delta $$ (like a click, or "revenue" made during the session etc.) on the recommended items. Traditionally, you'd generally create a usage matrix $$ U $$ where each cell would represent

This is my first website... I will be posting my thoughts on different Computer Science and Machine Learning articles.

Come back here when there's actual content.
