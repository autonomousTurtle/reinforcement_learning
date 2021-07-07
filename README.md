<!-- To develop equations, use: https://www.codecogs.com/latex/eqneditor.php and use the URL link to embed into markdown.-->


# Reinforcement Learning
Full repo to guided RL training scripts to learn dynamic programming, Q Learning, and Deep Q Learning

Resources: 

* [Nuts & Bolts of Reinforcement Learning: Model Based Planning using Dynamic Programming](https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/)
* [Reinforcement Learning Guide: Solving the Multi-Armed Bandit Problem from Scratch in Python](https://www.analyticsvidhya.com/blog/2018/09/reinforcement-multi-armed-bandit-scratch-python/?utm_source=blog&utm_medium=introduction-deep-q-learning-python)
* [A Hands-On Introduction to Deep Q-Learning using OpenAI Gym in Python](https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/)



## RL Key Equations and Concepts

### Markov Decision Process: 

![image](https://user-images.githubusercontent.com/31008838/124795723-57cd0900-df1e-11eb-91d1-1535a60c1c96.png)


### Sum of Rewards with Discount Factor: 

![SumWithDiscount](https://latex.codecogs.com/gif.latex?\LARGE&space;G_t\doteq&space;R_{t&plus;1}&space;&plus;&space;\gamma&space;R_{t&plus;2}&space;&plus;&space;\gamma^2&space;R_{t&plus;3}&plus;...=\sum_{k=0}^{\infty&space;}&space;\gamma^k&space;R_{t&plus;k&plus;1})


### State Value Function: How good is it to be in a given state?
*in other words, the average reward that the agent will get starting from the current state under policy pi*

![StateValueFunction](https://latex.codecogs.com/gif.latex?\large&space;v_{\pi&space;}(s)&space;\doteq&space;\mathbb{E}_{\pi}[G\displaystyle&space;_t&space;\mid&space;S_t&space;=&space;s]&space;=&space;\mathbb{E}_{\pi}&space;[\sum_{k=0}^{\infty}&space;\gamma^kR_{t&plus;k&plus;1}&space;\mid&space;S_t=s])  , for all  ![states](https://latex.codecogs.com/gif.latex?\inline&space;\large&space;s\epsilon&space;S)


### State-Action Value Function: How good an action is at a particular state?

![image](https://user-images.githubusercontent.com/31008838/124796804-91eada80-df1f-11eb-8a16-65154e2a4c53.png)

