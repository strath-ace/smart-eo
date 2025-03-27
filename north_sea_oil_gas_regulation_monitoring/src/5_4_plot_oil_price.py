from commons import *
import matplotlib.pyplot as plt


oil_dataset = csv_input("oil_price/Brent_oil_monthly_spot.csv")

# print(oil_dataset)

oil_price = []
trigger = False
for x, y in oil_dataset[1:]:
    if x == "15/01/2018":
        trigger = True
    if x == "15/01/2025":
        trigger = False
    if trigger:
        oil_price.append(float(y))


plt.figure(figsize=(9,4), layout="constrained")
plt.plot(oil_price)
plt.title("Brent Oil Spot Price Monthly")

xticker = np.arange(0,85,12)
xname = np.array(np.floor(xticker/12)+2018, dtype=int)
plt.xticks(xticker, xname)
plt.xlabel("Time (monthly)")
plt.ylabel("Brent Oil Spot Price ($ per barrel)")
plt.savefig("results/oil_price_simple.png",dpi=300)