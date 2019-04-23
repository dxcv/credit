from Loaders import Loaders

l = Loaders()
bonds = l.get_bond()
test = l.get_transboarder()
test = l.adf_coin_reg('transboarder', 'month', 1)
print(test)
