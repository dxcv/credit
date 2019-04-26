from Loaders import Loaders

l = Loaders()
bonds = l.get_bond()
test = l.get_strategy2()
test = l.adf_coin_reg('strategy2', 'month', 1)
print(test)
