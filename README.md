# Generative Adversarial Networks
> Presented here are some GAN architectures from my Master Thesis that generate MNIST and CelebA data and that solve Inpainting Problems.


Generative Adversarial Networks pose a powerful tool to learn distributions of data and are "the most interesting idea in machine learning since the last ten years" according to Yann LeCun https://medium.com/thecyphy/gans-what-and-where-b377672283c5. A Generative Adversarial Network (GAN) uses at its core the idea to let two neural networks compete against each other as players in a minimum-maximum (minmax or minimax) optimization game.\\
The game consists of two players, namely the generator $G$ and the discriminator $D$. For a given data set, the generator takes samples from a random probability distribution as input and produces fake data that is supposed to be as "similar" as possible to the real data. We will define later, what "similar" means. The discriminator takes as input both real and generated fake data and tries to discern, whether its input was indeed real or not. We can think of the generator as an art forger who tries to copy the paintings of an artist and we can think of the discriminator as the detective who examines the authenticity of both real and forged paintings. During the game, the forger and the detective become better at their function until the forger can produce perfect forgeries and the detective cannot tell anymore, whether the paintings are real or not. These players are usually represented by two neural networks and we train both the generator and the discriminator in an alternating fashion until they reach an equilibrium. The architecture of these networks is itself a hyper parameter or variable that can be tuned in order to achieve a better game. This is why a GAN should rather be thought of as a game theoretical concept. For the purpose of this thesis, we will assume feed-forward neural networks as \textit{multi-layer perceptrons} (MLP) if not otherwise stated.\\
GANs are widely used in the field of image generation. An impressive, state of the art GAN that has been trained sufficiently on human faces can be seen at\\ www.thispersondoesnotexist.com, which is based on the work of \cite{karras2020analyzing}. Lacking the same computational resources, we tried to accomplish a similar result which is showcased in Figure \ref{celebafaces}. In fact, GANs are not only limited to generating human faces, but are seemingly applicable to a limitless variety of data structures. This is showcased on the websites like www.thiscatdoesnotexist.com, www.thischemicaldoesnotexist.com or www.thismapdoesnotexist.com. It is well-known and confirmed by us, that GANs are volatile to the chosen hyperparameters and thus typically hard to train.

![](C:\Users\Karl\Documents\Arbeit\mt_pics\gen_celeba_tight.png)


## Usage example

A few motivating and useful examples of how your product can be used. Spice this up with code blocks and potentially more screenshots.

_For more examples and usage, please refer to the [Wiki][wiki]._




## Meta

Your Name – [@YourTwitter](https://twitter.com/dbader_org) – YourEmail@example.com

Distributed under the XYZ license. See ``LICENSE`` for more information.

[https://github.com/yourname/github-link](https://github.com/dbader/)


<!-- Markdown link & img dfn's -->
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki
