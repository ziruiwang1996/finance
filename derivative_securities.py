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
