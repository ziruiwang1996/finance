#swap rates
si = [7, 7.3, 7.7, 8.1, 8.4, 8.8] #spot rates
i = 1
sum_d_rate = 0
for each_spot_rate in si:
    discount_rate = 1/(1+each_spot_rate/100)**i
    i = i + 1
    sum_d_rate = sum_d_rate + discount_rate
#overall discount rate
d_overall = 1/(1+si[-1]/100)**len(si)
fixed_rate = (1-d_overall)/sum_d_rate
print('Fixed rate is', fixed_rate*100, '%')

#pricing European call(buy) option on 1-period binomial model
import numpy as np
S0=100 #stock price at t=0
R=1.02 #gross risk free rate
u=1.05 #stock up-move factor
d=1/u  #stock down-move factor
K = 102 #strike price
#payoffs for upper and lower bound price
if u*S0 > K :
    upper_option_payoff = u*S0 - K #exercise
else:
    upper_option_payoff = 0 #hold
if d*S0 > K :
    lower_option_payoff = d*S0 - K
else:
    lower_option_payoff = 0
#solving linear equations
A = np.matrix([[u*S0, R],[d*S0, R]])
B = np.matrix([[upper_option_payoff],[lower_option_payoff]])
A_inv = np.linalg.inv(A)
ans = A_inv * B
option_price = ans[0]*S0 + ans[1]*1
print('The fair value of the option is $', option_price)
