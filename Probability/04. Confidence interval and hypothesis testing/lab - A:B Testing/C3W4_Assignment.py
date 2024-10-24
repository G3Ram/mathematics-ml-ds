#!/usr/bin/env python
# coding: utf-8

# # AB Testing - Average Session Duration
# 
# Welcome! In this assignment you will be presented with two cases that require an AB test to choose an action to improve an existing product. You will perform AB test for a continuous and a proportion metric. For this you will define functions that estimate the relevant information out of the samples, compute the relevant statistic given each case and take a decision on whether to (or not) reject the null hypothesis.
# 
# Let's get started!
# 
# # Outline
# - [ 1 - Introduction](#1)
# - [ 2 - Exploring and handling the data](#2)
# - [ 3 - Revisiting the theory](#3)
# - [ 4 - Step by step computation](#4)
#   - [ Exercise 1](#ex01)
#   - [ Exercise 2](#ex02)
#   - [ Exercise 3](#ex03)
#   - [ Exercise 4](#ex04)
#   - [ Exercise 5](#ex05)
# 

# In[1]:


import math
import numpy as np
import pandas as pd
from scipy import stats


# In[2]:


import w4_unittest


# <a name="1"></a>
# ## 1 - Introduction
# 
# Suppose you have a website that provides machine learning content in a blog-like format. Recently you saw an article claiming that similar websites could improve their engagement by simply using a specific color palette for the background. Since this change seems pretty easy to implement you decide to run an AB test to see if this change does in fact drive your users to stay more time in your website.
# 
# The metric you decide to evaluate is the `average session duration`, which measures how much time on average your users are spending on your website. This metric currently has a value of 30.87 minutes.
# 
# Without further considerations you decide to run the test for 20 days by randomly splitting your users into two segments:
# - `control`: These users will keep seeing your original website.
# 
# 
# - `variation`: These users will see your website with the new background colors.
# 
# <a name="2"></a>
# ## 2 - Exploring and handling the data
# 
# Run the next cell to load the data from the test:

# In[3]:


# Load the data from the test using pd.read_csv
data = pd.read_csv("background_color_experiment.csv")

# Print the first 10 rows
data.head(10)


# In[4]:


print(f"The dataset size is: {len(data)}")


# The data shows for every user the average session duration and the version of the website they interacted with. To separate both segments for easier computations you can slice the Pandas dataframe by running the following cell. You may want to revisit our ungraded labs on Pandas if you are still not familiar with it. However, no need to worry because you don't need to be a Pandas expert to complete this assignment.

# In[5]:


# Separate the data from the two groups (sd stands for session duration)
control_sd_data = data[data["user_type"]=="control"]["session_duration"]
variation_sd_data = data[data["user_type"]=="variation"]["session_duration"]

print(f"{len(control_sd_data)} users saw the original website with an average duration of {control_sd_data.mean():.2f} minutes\n")
print(f"{len(variation_sd_data)} users saw the new website with an average duration of {variation_sd_data.mean():.2f} minutes")


# Notice that the split is not perfectly balanced. This is common in AB testing as there is randomness associated with the way the users are assigned to each group. 
# 
# At first glance it looks like the change to the background did in fact drive users to stay longer on your website. However you know better than driving conclusions at face value out of this data so you decide to perform a hypothesis test to know if there is a significant difference between the **means** of these two segments. 
# 
# <a name="3"></a>
# ## 3 - Revisiting the theory
# 
# Let's revisit the theory you saw in the lectures and apply it to this problem. If you are confident with the theory and you feel that you don't need a revision, you may skip this section direct to 1.4!
# 
# Remember that your job is to measure if changing the website's background color leads to an increase of the time visitors spend on it. Rewriting this as hypothesis test, the **null hypothesis** is that the change did not affect the time a visitor spend. Let's name the variables:
# 
# - $\mu_c$ is the average time a user **in the control group** spend in the website. Recall that the **control group** is the group accessing the website without the change in the background color.
# - $\mu_v$ is the average time a user **in the variation groups** spend in the website. Recall that the **variation group** is the groups accessing the website **with the updated background color**.
# 
# Also, recall that your intention is to measure if the background color leads to an **increase** in the time a visitor spend in the website. So writing this experiment as a hypothesis test, the **null hypothesis** is then $H_0: \mu_c = \mu_v$ and the **alternative hypothesis** is $H_1: \mu_v > \mu_c$, or equivalently, $H_1: \mu_v - \mu_c > 0$. 
# 
# Therefore, the hypothesis you will test is:
# 
# $$H_0: \mu_v = \mu_c \quad \text{vs.} \quad H_1: \mu_v - \mu_c > 0$$
# 
# As you saw in the lectures, this is a **right-tailed** test, as you are looking for an increase in the average time. As you saw above, you have more than 2000 users per group, this is a great amount of data so it is reasonable to rely in the Central Limit Theorem that the **average time** for each group follows a normal distribution. Remember that this result is for the group **average time** altogether and not that the time each user spend follows a normal distribution. You don't know the exact distribution for the amount of time a user spend in your website, however, the CLT assures that if we gather enough data, their average time will be very close to a normal distribution whose mean is the average time a user spend in the website. Let's then define two new quantities:
# 
# - $\overline{X}_c$ - the control group **sample mean**.
# - $\overline{X}_v$ - the variation group **sample mean**.
# - $n_c$ - the control group **size**.
# - $n_v$ - the variation group **size**.
# 
# So, by the Central Limit Theorem, you may suppose that
# 
# - $$\overline{X}_c \sim N\left(\mu_c, \left(\frac{\sigma_c}{\sqrt{n_c}}\right)^2\right)$$
# - $$\overline{X}_v \sim N\left(\mu_v, \left(\frac{\sigma_v}{\sqrt{n_v}}\right)^2\right)$$
# 
# Note that with our assumptions of normality, $\overline{X}_v - \overline{X}_c$ also follows a normal distribution. So, if $H_0$ is true, then $\mu_c = \mu_v$ and $\mu_v - \mu_c = 0$, therefore:
# 
# $$\overline{X}_c - \overline{X}_v \sim N\left(\mu_v - \mu_c, \left(\dfrac{\sigma_v}{\sqrt{n_v}}\right)^2 + \left(\dfrac{\sigma_c}{\sqrt{n_c}}\right)^2\right) = N\left(0, \left(\dfrac{\sigma_v}{\sqrt{n_v}}\right)^2 + \left(\dfrac{\sigma_c}{\sqrt{n_c}}\right)^2\right)$$
# 
# Or, equivalently:
# 
# $$\frac{\left( \overline{X}_v - \overline{X}_c \right)}{\sqrt{\left(\frac{\sigma_v}{\sqrt{n_v}}\right)^2 + \left(\frac{\sigma_c}{\sqrt{n_c}}\right)^2}} \sim N(0, 1)$$
# 
# However, remember that **you don't know the exact values for** $\sigma_v$ and $\sigma_c$, as they are the **population standard deviation** and you are working with a sample, so the best you can do is compute the **sample standard deviation**. So you must replace $\sigma_c$ and $\sigma_v$ by the sample standard deviation, respectively, $s_c$ and $s_v$. You also saw in the lectures that replacing the population standard deviation by the sample standard deviation changes the random variable from a Normal to a t-student:
# 
# $$t = \frac{\left( \overline{X}_v - \overline{X}_c \right)}{\sqrt{\left(\frac{s_v}{\sqrt{n_v}}\right)^2 + \left(\frac{s_c}{\sqrt{n_c}}\right)^2}} \sim t_d$$
# 
# Where $d$ is the **degrees of freedom** for this scenario. If we suppose that both groups have the same standard deviation, then $d = n_c + n_v - 2$, however there is no argument supporting this supposition, so the formula for the degrees of freedom gets a bit messier:
# 
# $$d = \frac{\left[\frac{s_{v}^2}{n_v} + \frac{s_{c}^2}{n_c} \right]^2}{\frac{(s_{v}^2/n_v)^2}{n_v-1} + \frac{(s_{c}^2/n_c)^2}{n_c-1}}$$
# 
# Once you get the actual value for $t_d$ the, with a given significance level $\alpha$, you can decide if this value falls within the range of values that are likely to occur in the $t$-student distribution (where 'likely' is related with your significance level). To perform this step you must find the value $p$ such that 
# 
# $$p = P(t_d > t | H_0)$$
# 
# If this value is less than your significance level $\alpha$, then you **reject the null hypothesis**, because it means that you observed a value that is very unlikely to occur (unlikely here means that is less than the significance level you have set) if $H_0$ is true.
# 
# Also, remember that $P(t_d \leq t)$ is the $\text{CDF}$ (cumulative distribution function) for the $t$-student distribution with $d$ degrees of freedom in the point $x = t$, so to compute $P(t_d > t)$ you may compute:
# 
# $$P(t_d > t) = 1 - \text{CDF}_{t_d}(t)$$
# 
# Since $P(t_d \leq t) + P(t_d > t) = 1$

# <a name="4"></a>
# ## 4 - Step by step computation
# 
# 
# Wrapping up everything discussed above:
# 
# The hypothesis test is given by:
# 
# $$H_0: \mu_v = \mu_c \quad \text{vs.} \quad H_1: \mu_v - \mu_c > 0$$
# 
# You will start computing:
# 
# - $n_c$ and $n_v$, the control and variation group sizes, respectively.
# - $\overline{X}_c$ and $\overline{X}_v$, the average time spent by the users in the control and variation group, respectively. 
# - $s_c$ and $s_v$, the **sample** standard deviation for the time spend by the users in the control and variation group, respectively.
# 
# With these quantities in hand, the next steps are to compute:
# 
# - $d$, the degrees of freedom of the $t$-student distribution, $t_d$.
# - The $t$-value, which it will be called $t$.
# - The $p$ value for the distribution $t_d$ for the $t$-value, i.e., the value  $p = P(t_d > t | H_0)$.
# 
# Finally, for a given significance level $\alpha$, you will be able to decide if you reject or not $H_0$, depending on wether $p \leq \alpha$ or not.
# 
# Let's get your hands into work now! Run the cell below to retrieve the session times for the control and variation groups.

# In[6]:


# X_c stores the session tome for the control group and X_v, for the variation group. 
X_c = control_sd_data.to_numpy()
X_v = variation_sd_data.to_numpy()


# In[7]:


print(f"The first 10 entries for X_c are:\n{X_c[:20]}\n")
print(f"The first 10 entries for X_v are:\n{X_v[:20]}\n")


# <a name="ex01"></a>
# ### Exercise 1
# 
# In this exercise, you will write a function to retrieve the basic statistics for `X_c` and `X_d`. In other words, this function will compute, for a given numpy array:
# 
# - Its size (in your case, $n_c$ and $n_v$).
# - Its mean (in your case, $\overline{X}_c$ and $\overline{X}_v$)
# - Its sample standard deviation(in your case, $s_c$ and $s_v$)
# 
# This function inputs a numpy array and outputs a tuple in the form `(n, x, s)` where `n` is the numpy array size, `x` is its mean and `s`is its **sample** standard deviation.
# 
# Hint: 
# - Recall that the sample standard deviation is computed by replacing $N$ by $N-1$ in the variance formula. 
# - You may compute an array size using the `len`function.
# - Any array in numpy has a method called `.mean()` to compute its mean.
# - Any array in numpy has a method called `.std()` to compute the standard deviation and a parameter called `ddof` where if you pass `ddof = 1`, it will use $N-1$ instead of $N$. 

# In[8]:


def get_stats(X):
    """
    Calculate basic statistics of a given data set.

    Parameters:
    X (numpy.array): Input data.

    Returns:
    tuple: A tuple containing:
        - n (int): Number of elements in the data set.
        - x (float): Mean of the data set.
        - s (float): Sample standard deviation of the data set.
    """

    ### START CODE HERE ###
    
    # Get the group size
    n = len(X)
    # Get the group mean
    x = np.mean(X)
    # Get the group sample standard deviation (do not forget to pass the parameter ddof if using the method .std)
    s = np.std(X, ddof=1)

    ### END CODE HERE ###

    return (n,x,s)


# In[9]:


w4_unittest.test_get_stats(get_stats)


# In[10]:


n_c, x_c, s_c = get_stats(X_c)
n_v, x_v, s_v = get_stats(X_v)


# In[11]:


print(f"For X_c:\n\tn_c = {n_c}, x_c = {x_c:.2f}, s_c = {s_c:.2f} ")
print(f"For X_v:\n\tn_v = {n_v}, x_v = {x_v:.2f}, s_v = {s_v:.2f} ")


# ##### __Expected Output__ 
# 
# ```Python
# For X_c:
# 	n_c = 2069, x_c = 32.92, s_c = 17.54 
# For X_v:
# 	n_v = 2117, x_v = 33.83, s_v = 18.24 
# ```

# <a name="ex02"></a>
# ### Exercise 2
# 
# In this exercise you will implement a function to compute $d$, the degrees of freedom for the $t$-student distribution. It is given by the following formula:
# 
# $$d = \frac{\left[\frac{s_{c}^2}{n_c} + \frac{s_{v}^2}{n_v} \right]^2}{\frac{(s_{c}^2/n_c)^2}{n_c-1} + \frac{(s_{v}^2/n_v)^2}{n_v-1}}$$
# 
# Hint: You may use the syntax `x**2`to square a number in python, or you may use the function `np.square`. The latter may help to keep your code cleaner. Pay attention in the parenthesis as they will indicate the order that Python will perform the computation!

# In[19]:


def degrees_of_freedom(n_v, s_v, n_c, s_c):
    """Computes the degrees of freedom for two samples.

    Args:
        control_metrics (estimation_metrics_cont): The metrics for the control sample.
        variation_metrics (estimation_metrics_cont): The metrics for the variation sample.

    Returns:
        numpy.float: The degrees of freedom.
    """
    
    ### START CODE HERE ###
    
    # To make the code clean, let's divide the numerator and the denominator. 
    # Also, note that the value s_c^2/n_c and s_v^2/n_v appears both in the numerator and denominator, so let's also compute them separately

    # Compute s_v^2/n_v (remember to use Python syntax or np.square)
    s_v_n_v = (np.square(s_v))/n_v

    # Compute s_c^2/n_c (remember to use Python syntax or np.square)
    s_c_n_c = (np.square(s_c))/n_c


    # Compute the numerator in the formula given above
    calc1 = s_v_n_v + s_c_n_c
    numerator = np.square(calc1)

    # Compute the denominator in the formula given above. Attention that s_c_n_c and s_v_n_v appears squared here!
    # Also, remember to use parenthesis to indicate the operation order. Note that a/b+1 is different from a/(b+1).
    denominator = ((np.square(s_c_n_c))/(n_c - 1)) + ((np.square(s_v_n_v))/(n_v - 1))
    
    ### END CODE HERE ###

    dof = numerator/denominator
        
    return dof


# In[20]:


w4_unittest.test_degrees_of_freedom(degrees_of_freedom)


# In[21]:


d = degrees_of_freedom(n_v, s_v, n_c, s_c)
print(f"The degrees of freedom for the t-student in this scenario is: {d:.2f}")


# ### __Expected output__
# 
# `The degrees of freedom for the t-student in this scenario is: 4182.97
# `

# <a name="ex03"></a>
# ### Exercise 3
# 
# In this exercise, you will compute the $t$-value, given by
# 
# $$t = \frac{\left( \overline{X}_v - \overline{X}_c \right)}{\sqrt{\left(\frac{s_v}{\sqrt{n_v}}\right)^2 + \left(\frac{s_c}{\sqrt{n_c}}\right)^2}} = \frac{\left( \overline{X}_v - \overline{X}_c \right)}{\sqrt{\frac{s_v^2}{n_v} + \frac{s_c^2}{n_c}}}$$
# 
# Remember that you are storing $\overline{X}_c$ and $\overline{X}_v$ in the variables $x_c$ and $x_d$, respectively. 

# In[22]:


def t_value(n_v, x_v, s_v, n_c, x_c, s_c):

    ### START CODE HERE ###

    # As you did before, let's split the numerator and denominator to make the code cleaner.
    # Also, let's compute again separately s_c^2/n_c and s_v^2/n_v.

    # Compute s_v^2/n_v (remember to use Python syntax or np.square)
    s_v_n_v = (np.square(s_v))/n_v

    # Compute s_c^2/n_c (remember to use Python syntax or np.square)
    s_c_n_c = (np.square(s_c))/n_c

    # Compute the numerator for the t-value as given in the formula above
    numerator = x_v - x_c

    # Compute the denominator for the t-value as given in the formula above. You may use np.sqrt to compute the square root.
    dp1 = np.square(s_v/(np.sqrt(n_v)))
    dp2 = np.square(s_c/(np.sqrt(n_c)))
    denominator = np.sqrt(dp1 + dp2)
    
    ### END CODE HERE ###

    t = numerator/denominator

    return t


# In[23]:


w4_unittest.test_t_value(t_value)


# In[24]:


t = t_value(n_v, x_v, s_v, n_c, x_c, s_c)
print(f"The t-value for this experiment is: {t:.2f}")


# ##### __Expected Output__
# 
# ```
# The t-value for this experiment is: 1.64
# ```

# <a name="ex04"></a>
# ### Exercise 4
# 
# In this exercise, you will compute the $p$ value for $t_d$, for a given significance level $\alpha$. Recall that this experiment is a right-tailed t-test, because you are investigating wether the background color change increases the time spent by users in your website or not. 
# 
# In this experiment the $p$-value for a significance level of $\alpha$ is given by
# 
# $$p = P(t_d > t) = 1 - \text{CDF}_{t_d}(t)$$
# 
# 
# Hint: 
# - You may use the scipy function `stats.t(df = d)` to get the $t$-student distribution with `d`degrees of freedom. 
# - To compute its CDF, you may use its method `.cdf`. 
# 
# Example:
# 
# Suppose you want to compute the CDF for a $t$-student distribution with $d = 10$ degrees of freedom for a t-value of $1.21$.

# In[25]:


t_10 = stats.t(df = 10)
cdf = t_10.cdf(1.21)
print(f"The CDF for the t-student distribution with 10 degrees of freedom and t-value = 1.21, or equivalently P(t_10 < 1.21) is equal to: {cdf:.2f}")


# This means that there is a probability of 87% that you will observe a value less than 1.21 when sampling from a $t$-student distribution with 10 degrees of freedom.
# 
# Ok, now you are ready to write a function to compute the $p$-value for the $t$-student distribution, with $d$ degrees of freedom and a given $t$-value.

# In[31]:


def p_value(d, t_value):

    ### START CODE HERE ###

    # Load the t-student distribution with $d$ degrees of freedom. Remember that the parameter in the stats.t is given by df.
    t_d = stats.t(df = d)

    # Compute the p-value, P(t_d > t). Remember to use the t_d.cdf with the proper adjustments as discussed above.
    p = 1 - t_d.cdf(t_value)


    ### END CODE HERE ###

    return p


# In[32]:


w4_unittest.test_p_value(p_value)


# In[33]:


print(f"The p-value for t_15 with t-value = 1.10 is: {p_value(15, 1.10):.4f}")
print(f"The p-value for t_30 with t-value = 1.10 is: {p_value(30, 1.10):.4f}")


# ### __Expected output__
# 
# ```
# The p-value for t_15 with t-value = 1.10 is: 0.1443
# The p-value for t_30 with t-value = 1.10 is: 0.1400
# ```

# <a name="ex05"></a>
# ### Exercise 5
# 
# In this exercise you will wrap up all the functions you have built so far to decide if you accept $H_0$ or not, given a significance level of $\alpha$.
# 
# It will input both control and validation groups and it will output `Reject H_0$` or `Do not reject H_0` accordingly.
# 
# Remember that you **reject** $H_0$ if the p-value is **less than** $\alpha$. 

# In[34]:


def make_decision(X_v, X_c, alpha = 0.05):

    ### START CODE HERE ###

    # Compute n_v, x_v and s_v
    n_v, x_v, s_v = get_stats(X_v)

    # Compute n_c, x_c and s_c
    n_c, x_c, s_c = get_stats(X_c)

    # Compute the degrees of freedom for the t-student distribution for this experiment.
    # Pay attention to the arguments order. You may look the function definition above to make sure you don't swap values.
    # Also, remember that x_c and x_v are not used in this computation
    d = degrees_of_freedom(n_v, s_v, n_c, s_c)
    
    # Compute the t-value
    t = t_value(n_v, x_v, s_v, n_c, x_c, s_c)

    # Compute the p-value for the t-student distribution with d degrees of freedom
    p = p_value(d, t)

    # This is the decision step. Compare p with alpha to decide about rejecting H_0 or not. 
    # Pay attention to the return value for each block to properly write the condition.

    if p < alpha:
        return 'Reject H_0'
    else:
        return 'Do not reject H_0'

    ### END CODE HERE ###


# In[35]:


w4_unittest.test_make_decision(make_decision)


# In[36]:


alphas = [0.06, 0.05, 0.04, 0.01]
for alpha in alphas:
    print(f"For an alpha of {alpha} the decision is to: {make_decision(X_v, X_c, alpha = alpha)}")


# **Congratulations on finishing this assignment!**
# 
# Now you have created all the required steps to perform an AB test for a simple scenario!
# 
# **This is the last assignment of the course and the specialization so give yourself a pat on the back for such a great accomplishment! Nice job!!!!**

# 

# In[ ]:




