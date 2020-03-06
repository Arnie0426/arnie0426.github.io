---
layout: post
title:  "A Case For Embeddings In Recommendation Problems"
date:   2020-03-05 06:00
categories: [embeddings, deep-learning, recommender-systems]
comments: false
---

<meta name="twitter:description" content = "Once you have worked on different machine learning problems, most things in the field start to feel very similar. You take your raw input data, map it to a different latent space with fewer dimensions, and then perform your classification/regression/clustering. Recommender systems, new and old, are no different. In the classic collaborative filtering problem, you factorize your partially filled usage matrix to learn *user-factors* and *item-factors*, and try to predict user ratings with a dot-product of the factors."/>
<meta property="og:description" content = "Once you have worked on different machine learning problems, most things in the field start to feel very similar. You take your raw input data, map it to a different latent space with fewer dimensions, and then perform your classification/regression/clustering. Recommender systems, new and old, are no different. In the classic collaborative filtering problem, you factorize your partially filled usage matrix to learn *user-factors* and *item-factors*, and try to predict user ratings with a dot-product of the factors."/>
<meta property="og:title" content = "A Case For Embeddings In Recommendation Problems"/>
<meta property="og:image" content = "{{ site.url }}/assets/MF.png"/>
<meta name="twitter:image" content = "{{ site.url }}/assets/MF.png"/>

Once you have worked on different machine learning problems, most things in the field start to feel very similar. You take your raw input data, map it to a different latent space with fewer dimensions, and then perform your classification/regression/clustering. Recommender systems, new and old, are no different. In the [classic collaborative filtering](https://www.benfrederickson.com/matrix-factorization/) problem, you factorize your partially filled usage matrix to learn *user-factors* and *item-factors*, and try to predict user ratings with a dot-product of the factors.

<img src="/assets/MF.png" alt="Matrix Factorization"  style="padding-left: 10%; padding-right: 10%; text-align:center;">

<!--more-->

This has worked well for many people at different companies and I have also had successes with it firsthand at [Flipboard](http://flipboard.com). And of course, people try to incorporate [more signals](https://cs.stanford.edu/people/jure/pubs/reviews-recsys13.pdf) into this model to get better performance for [cold-start](https://en.wikipedia.org/wiki/Cold_start_(recommender_systems)) and other [domain specific problems](https://www.youtube.com/watch?v=RtCiGhnskcM).

However, I didn't really care about using fancy deep learning techniques for my recommendation problems until a [friend](https://ivesmacedo.com) asked me a very simple question at a [conference](https://recsys.acm.org/) a few years ago. If I recall correctly, he questioned my use of a certain [regularizer](https://en.wikipedia.org/wiki/Regularization_(mathematics)), and I soon realized that his clever suggestion required me to go back to the whiteboard, recompute all the gradients and optimization steps, and essentially reimplement the core algorithm from scratch to test a relatively straightforward modification - I wasn't writing [PyTorch](https://pytorch.org/) code, I only wished I did.

### Enter AutoGrad and Embeddings

So, as it turns out the classic matrix-factorization problem can be formulated as a deep learning problem if you just think of the *user-factors* and *item-factors* as *embeddings*. An embedding is simply a mapping of a discrete valued list to a real valued lower dimensional vector (*cough*). Looking at the problem from this perspective gives you a lot more modelling flexibility thanks to the number of great autograd software out there. If you randomly initialize these embeddings, and define [mean-squared-error](https://en.wikipedia.org/wiki/Mean_squared_error) as your loss, [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) would get you embeddings that would be very similar to what you would get with matrix factorization.

<img src="/assets/FFN.png" alt="Feed-forward view of RecSys"  style="padding-left: 10%; padding-right: 10%; text-align:center;">

But as [Justin Basilico](https://twitter.com/JustinBasilico) showed in his informative ICML workshop [talk](https://www.slideshare.net/justinbasilico/recent-trends-in-personalization-a-netflix-perspective), modelling the problem as a deep feed-forward network makes the learning task a lot trickier. Due to having more parameters and hyper-parameters, it requires more compute while only providing [questionable improvements](https://arxiv.org/abs/1907.06902) for the actual task. So why should we bother thinking of the problem in this way?

I would argue that modelling flexibility and experimentation ease are nothing to be scoffed at. This perspective allows you to incorporate [all](https://arxiv.org/pdf/1607.07326.pdf) [sorts](https://openreview.net/pdf?id=ryTYxh5ll) of [data](https://arxiv.org/abs/1510.01784) into this framework fairly easily. Recommendation is also more than just predicting user-ratings, and you can solve many other recommendation problems such as [sequence-aware recommendations](https://arxiv.org/abs/1802.08452) a lot easier. Not to mention, because of autograd software, you end up with much shorter code that allows you to tweak things a lot quicker. I like [optimizing my matrix-factorization with conjugate gradient](https://www.benfrederickson.com/fast-implicit-matrix-factorization/) as much as everyone else but please don't ask me to recompute my CG steps after you add some new data and change your regularizer in the year 2020. 

### Other Ways To Learn Embeddings

The other great thing about embeddings is that there are several different ways of learning this mapping. If you don't want to learn embeddings through random initialization and backpropagation from an input matrix, one very common approach is *[Skip-gram with negative-sampling](https://arxiv.org/abs/1310.4546)*. This method has been [extremely popular](https://github.com/google-research/bert) in natural language processing, and has also been successful in creating embeddings from non-textual sequences such as *[graph-nodes](https://snap.stanford.edu/node2vec/)*, *[video games](https://arxiv.org/abs/1603.04259)* and *[Pinterest pins](https://medium.com/the-graph/applying-deep-learning-to-related-pins-a6fee3c92f5e)*.

The core idea in *skip-gram with negative sampling* is to create a dataset with *positive* examples by sliding a context-window through a sequence and creating pairs of items that co-occur with a central item, and also generating *negative* data by random sampling items from the entire corpus, and create pairs of items that do not usually co-occur in the same window.

<img src="/assets/animation1.gif" alt="Skipgram"  style="padding-left: 10%; padding-right: 10%; text-align:center;">

Once you have the dataset with both postive and negative examples, you simply train a classifier with a deep neural network and [learn your embeddings](https://cs224d.stanford.edu/lecture_notes/notes1.pdf). In this formulation, things that co-occur close to each other would have similar embeddings, which is usually what we need for most search and recommendation tasks.

For recommendation, there are many different ways to create these sequences. Airbnb has a [great paper](https://www.kdd.org/kdd2018/accepted-papers/view/real-time-personalization-using-embeddings-for-search-ranking-at-airbnb) on how they collect sequences of listings based on a user's sequential clicks of listings during a search/booking session to learn item-embeddings. Alibaba has another [interesting way](https://arxiv.org/pdf/1803.02349.pdf) where they maintain an item-item interaction graph, where an edge from an item A to B indicates how often a user clicked on an item B after an item A, and then use random walks in the graph to generate sequences.

### So What's So Cool About These Embeddings?

In addition to the task at hand that each of those representations help solve (such as [finding similar items](https://medium.com/the-graph/applying-deep-learning-to-related-pins-a6fee3c92f5e)), they are modular and amenable for *[transfer learning](https://en.wikipedia.org/wiki/Transfer_learning)*. One great thing about deep learning has been that you *almost never* have to start solving a problem from scratch, and all these different embeddings act as great places to start for new problems. If you wanted to build a new classifier (say, a spam detector), you could use your item embeddings as a starting point, and would be able to train a model much quicker with some basic [fine-tuning](http://wiki.fast.ai/index.php/Fine_tuning). 

These modular mappings to latent spaces have been extremely useful for me, and in addition to solving some recommendation problems, I have also been able to reuse and fine-tune these embeddings and solve many different end-tasks. In addition, storing these embeddings in a centralized [model storage](https://mlflow.org/docs/latest/models.html) also helps teams reduce redundancy and provides them with good foundations to build on for many problems. 

While I hadn't initially bought into the whole deep learning for recommender systems craze, I am starting to see beyond just the minimal performance gains on the original task, and highly recommend everyone to play around with this (still relatively new) paradigm in recommender systems!
