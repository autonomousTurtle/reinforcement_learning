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

<!-- + is %2B, - is %2D -->
<img src="https://render.githubusercontent.com/render/math?math=\G_t= R_t_%2B_1 %B R_t + 2 + R_t + 3 ... + R_T">

![image](https://user-images.githubusercontent.com/31008838/124796227-ea6da800-df1e-11eb-87c6-c1eb2c35143c.png)


### State Value Function: How good is it to be in a given state?
*in other words, the average reward that the agent will get starting from the current state under policy pi*

![image](https://user-images.githubusercontent.com/31008838/124796462-30c30700-df1f-11eb-8cb0-c9efd3d08ff3.png)


### State-Action Value Function: How good an action is at a particular state?

![image](https://user-images.githubusercontent.com/31008838/124796804-91eada80-df1f-11eb-8a16-65154e2a4c53.png)

