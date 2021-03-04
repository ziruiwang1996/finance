
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
    #expiration: contract expirate in ? (time unit match with compoun term)
    d = 1/((1+r/interest_compound_term)**(expiration/(12/interest_compound_term)))
    return d
s0 = 400 #current stock price per share
forward_price = s0/discount_rate(0.08, 4, 9)
print(forward_price)

#value of forward contract at an intermediate time
#def forward_price (si, st):
S0 = 100.0
St = 125.0
r = 0.1
n = 2
F0 = S0/d(0, 1, r, n)
#ft=(Ft-F0)*d(t, T)
ft = St-F0*d(0.5, 1, r, n)
print('Q7: {:0.1f}'.format(round(ft, 1)))
