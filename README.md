# TOSCA-ML

TOSCA-ML is an automated machine learning implementation using TOSCA specification. It allows you to read data, preprocess it, train different ML models, and evaluate them. The implementation is built using a reusable components approach, which are connected with the help of various relationships. TOSCA-ML may not reuse all the models developed under the [Radon-Particle](https://github.com/radon-h2020/radon-particles) project. The RADON Organisation can be found here: [https://github.com/radon-h2020](https://github.com/radon-h2020).

### Why use TOSCA-ML?

You can imagine a situation when you want to have a small experiment or research without coding experience. Or, for example, you know exactly how to code, but you do not want to start from scratch and do it as fast as possible. Or the worst scenario, in which the working computer does not have the necessary capacity to execute the whole project script. Given project solves those problems. Having at least some understanding of data science and the needed steps for an excellent machine-learning model is enough. You must assemble logical components, and the orchestrator will do everything for you.

### Getting started

First, you must install xOpera orchestrator, which will execute TOSCA scripts.

```bash
pip install opera
```

After that, you must install [Eclipse Winery's](https://winery.readthedocs.io/en/latest/index.html) web-based environment, allowing you to manipulate and build an automated machine learning pipeline.

```bash
docker run -it -p 8080:8080 \
  -e PUBLIC_HOSTNAME=localhost \
  -e WINERY_FEATURE_RADON=true \
  -e WINERY_REPOSITORY_PROVIDER=yaml \
  -v <path_on_your_host>:/var/repository \
  -u `id -u` \
  opentosca/radon-gmt
```

That is it! Now you can visit http://localhost:8080/ and start building a machine learning pipeline. The famous data science Titanic project has also been solved, which can be found under the _Service Templates_ called _titanic_classification_problem_. It may give some hints and understanding of how the pipeline can be built and how it may look. The descriptive image is also shown below.

![mltosca_titanic](https://user-images.githubusercontent.com/22376543/233140120-5eafad79-c3fa-46fa-ad63-678157869869.png)

This template can also be downloaded and executed

```bash
xopera deploy titanic_classification_problem.csar
```

### Understanding service template

Looking at the image above, you can understand that the shown service template is a directed acyclic graph. All paths lead to the node, which is called _Conda_. It is a mandatory fundament on which all other components will be hosted. After that, the data reading component is executed. Gained data will be passed to the preprocessing part consisting of multiple parts executed in a specified order. Next, preprocessed data is split and passed to four machine learning models. Finally, trained models are evaluated.

It can be noticed that even though arrows show flow from right to left, the flow is executed in the opposite direction. The reason is that relationship _DependsOn_ tells that the component depends on the specific node and will not run before that. In other words, this connection presents the execution order. There is another relationship called _HostedOn_, which connects nodes to the whole template and specifies child components link to the parents.
