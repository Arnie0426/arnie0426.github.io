---
layout: post
title:  "Why Embeddings Are Great for Recommendation Problems"
date:   2020-03-01 02:19
categories: [deep-learning, recommender-systems]
comments: false
---

<meta name="twitter:description" content = "When you take a step back, most things in machine learning feel very similar. You take your raw input data, map it to a different latent space with fewer dimensions, and then perform your classification/regression/clustering. Recommender systems, new and old, are no different. In the classic collaborative filtering problem, you learn user-factors and item-factors by factorizing your partially filled usage matrix and try to predict whether a user would like a certain item."/>
<meta property="og:description" content = "When you take a step back, most things in machine learning feel very similar. You take your raw input data, map it to a different latent space with fewer dimensions, and then perform your classification/regression/clustering. Recommender systems, new and old, are no different. In the classic collaborative filtering problem, you learn user-factors and item-factors by factorizing your partially filled usage matrix and try to predict whether a user would like a certain item."/>

When you take a step back, most things in machine learning feel very similar. You take your raw input data, map it to a different latent space with fewer dimensions, and then perform your classification/regression/clustering. 

Recommender systems, new and old, are no different. In the [classic collaborative filtering](https://www.benfrederickson.com/matrix-factorization/) problem, you learn user-factors and item-factors in a latent space by factorizing your partially filled usage matrix and try to predict whether a user would like a certain item.

<img src="/assets/MF.png" alt="Modern ML"  style="padding-left: 10%; padding-right: 10%; text-align:center;">

<!--more-->

This has worked well for many different companies and industries as well as at my day job at [Flipboard](http://flipboard.com). And of course, people try to incorporate [more signals](https://cs.stanford.edu/people/jure/pubs/reviews-recsys13.pdf) into this model to get better performance for [cold-start](https://en.wikipedia.org/wiki/Cold_start_(recommender_systems)) and other [domain specific problems](https://www.youtube.com/watch?v=RtCiGhnskcM). 

It wasn't until a [friend](https://ivesmacedo.com) at a [conference](https://recsys.acm.org/) a few years ago asked me a very simple question that I didn't care about using fancy deep learning techniques for my recommendation problems. If I recall correctly, he questioned my use of a certain [regularizer](https://en.wikipedia.org/wiki/Regularization_(mathematics)), and I soon realized that his clever suggestion required me to go back to the whiteboard, redesign and recompute all the gradients and optimization steps, and essentially reimplement the core algorithm from scratch to test such a simple modification - I wasn't writing [PyTorch](https://pytorch.org/) code, I only wished I did.

## Enter AutoGrad and Embeddings

So, as it turns out the classic matrix-factorization problem can be formulated as a deep learning problem if you just think of the *user-factors* and *item-factors* as *embeddings*. An embedding is simply a mapping of a discrete valued list to a real valued lower dimensional space (*cough*). Looking at the problem from this perspective gives you a lot more modelling flexibility thanks to the number of great auto-grad software out there. 

<img src="/assets/FFN.png" alt="Modern ML"  style="padding-left: 10%; padding-right: 10%; text-align:center;">

But as [Justin Basilico](https://twitter.com/JustinBasilico) showed in his wonderful ICML workshop [talk](https://www.slideshare.net/justinbasilico/recent-trends-in-personalization-a-netflix-perspective), modelling the problem as a deep feed-forward network makes the optimization task a lot trickier. Due to having more parameters and hyper-parameters, it requires more compute while only providing [questionable improvements](https://arxiv.org/abs/1907.06902) for the actual task. So why should we bother?

I would argue that modelling flexibility and experimentation ease are nothing to be scoffed at. It allows you to incorporate [all](https://arxiv.org/pdf/1607.07326.pdf) [sorts](https://openreview.net/pdf?id=ryTYxh5ll) of [data](https://arxiv.org/abs/1510.01784) into this framework fairly easily. In addition to this, you can solve many different recommendation problems such as [sequence-aware recommendation](https://arxiv.org/abs/1802.08452) a lot easier. 

## Other ways to learn embeddings

The other great thing about embeddings is that there are several different ways of learning this mapping. If you don't want to learn embeddings through random initialization and backpropagation from an input matrix, one very common approach is *[Skip-gram with negative-sampling](https://arxiv.org/abs/1310.4546)*. This method has been extremely popular in the natural language processing world, and has also been successful in creating embeddings from non-textual sequences such as *[products](https://arxiv.org/pdf/1803.02349.pdf)*, *[graph-nodes](https://snap.stanford.edu/node2vec/)*, *[video games](https://arxiv.org/abs/1603.04259)* and *[Airbnb listings](https://www.kdd.org/kdd2018/accepted-papers/view/real-time-personalization-using-embeddings-for-search-ranking-at-airbnb)*.

The core-idea in *skip-gram with negative sampling* is to create a *positive* data-set by sliding a context-window through a sequence and creating pairs of items that co-occur with the central word, and also generating and a *negative* data-set by random sampling items from the entire corpus, and create pairs of items that do not usually co-occur in the same window.

[Animation]

Once you have both datasets, you simply train a classifier with a deep neural network and learn your embeddings. In this formulation, things that co-occur close to each other would have similar embeddings, which is usually what we need for most search and recommendation tasks.

## So what's so cool about them?

In addition to the task at hand that each of those representations help solve ([finding similar items](https://medium.com/the-graph/applying-deep-learning-to-related-pins-a6fee3c92f5e)), they also act as great places to start for completely different tasks for *[transfer learning](https://en.wikipedia.org/wiki/Transfer_learning)*. 

One great thing about deep learning has been that you *almost* never have to start solving a problem from scratch, and all these different embeddings act as great places to start for newer problems. If you wanted to build a click-bait detector, you could use your item embeddings as a starting point, and would be able to train a model much quicker  I have personally found it extremely useful to have a [model storage](https://mlflow.org/docs/latest/models.html) and trying different embeddings (or concatenation of everything!) as a starting point for my tasks at hand. Having modular mappings to latent spaces that I can mostly re-use and fine-tune for my end-tasks has been an absolute game changer for me! 

