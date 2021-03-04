#present value calculation
def present_value (r, ci, period):
    # r: interest rate
    # ci:cash flow per future period
    pv = 0
    for i in range(0, period):
        pvi = ci/(1+r)**i
        pv = pv + pvi
    return pv
print(present_value(0.1, 500000, 20))

#forward contract on a stock
def discount_rate (r, interest_compound_term, expiration):
    #r: spot rate
    #interest_compound_term e.g. annually=1, quarterly=4
    #expiration: contract expire in ? (time unit match with compound term)
    d = 1/((1+r/interest_compound_term)**(expiration/(12/interest_compound_term)))
    return d
s0 = 400 #current stock price per share
forward_price = s0/discount_rate(0.08, 4, 9)
print(forward_price)

#value of forward contract at an intermediate time
def forward_price (Si, r, interest_compound_term, expiration):
    Fi = Si/discount_rate(r, interest_compound_term, expiration)
    return Fi
F0 = forward_price(100, 0.1, 2, 12) #half year ago to half year later
Ft = forward_price(125, 0.1, 2, 6)  #now to half year later
ft = (Ft-F0)*discount_rate(0.1, 2, 6)
print('Current value of forward contract is $', ft)
