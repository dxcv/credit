import time
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
from sklearn.metrics import r2_score
import gc
import copy
import re
import os
from WindPy import *
w.start()


class Loaders:
    def __init__(self):
        self.prospect = ['M0017126', 'M0017127', 'M0017128', 'M0017129', 'M0017130', 'M0017131', 'M0017132',
                         'M0017133', 'M5766711', 'M0017134', 'M0017135', 'M0017136', 'M0017137', 'M5207790',
                         'M0000138', 'M0061603', 'M0290204']
        self.prospect_name = ['PMI', 'PMI:生产', 'PMI:新订单', 'PMI:新出口订单', 'PMI:在手订单', 'PMI:产成品库存',
                              'PMI:采购量', 'PMI:进口', 'PMI:出厂价格', 'PMI:主要原材料购进价格', 'PMI:原材料库存',
                              'PMI:从业人员', 'PMI:供货商配送时间', 'PMI:生产经营活动预期', '财新中国PMI',
                              '财新中国服务业PMI:经营活动指数', '财新中国综合PMI:产出指数']

    def get_prospect(self):
        prospect_code = ""
        now_date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        for code in self.prospect:
            prospect_code += code
        raw_data = w.edb(prospect_code, "2010-01-01", now_date)
        data = pd.DataFrame()
        data['date'] = pd.Series(raw_data.Times)
        for i in range(len(raw_data.Codes)):
            data[self.prospect_name[i]] = pd.Series(raw_data.Data[i])
        data.set_index(['date'], inplace=True)
        return data

    @staticmethod
    def ADFtest(data):
        """
        平稳性检验
        :param data:
        :return: dateframe of results
        """
        alls = {'时间序列(差分处理)': [],
                'ADF值': [],
                '1%临界值': [],
                '5%临界值': [],
                '是否平稳': []}
        for col in data.columns:
            alls['时间序列(差分处理)'].append(col)
            data_rmna = data[col].dropna().diff(1).dropna()
            adftest = adfuller(data_rmna, autolag='AIC')
            alls['ADF值'].append(adftest[0])
            alls['1%临界值'].append(adftest[4]['1%'])
            alls['5%临界值'].append(adftest[4]['5%'])
            if adftest[0] < adftest[4]['1%']:
                alls['是否平稳'].append('平稳')
            else:
                alls['是否平稳'].append('非平稳')
        alls = pd.DataFrame.from_dict(alls)
        alls = alls[['时间序列(差分处理)', 'ADF值', '1%临界值', '5%临界值', '是否平稳']]
        return alls

    # @staticmethod
    # def cointest(data, period):
    #     """
    #     协整检验
    #     :param data:
    #     :param period:
    #     :return:
    #     """
    #     alls = {'指标变化率的时间序列': [],
    #            'P值': [],
    #            '是否协整': []}
    #     if period == 'special day':
    #         for col in data.columns:
    #             target = y.join(data[col], how='outer')
    #             target[name].fillna(inplace=True, method='ffill', limit=31)
    #             target.dropna(inplace=True)
    #             cotest = coint(target[col], target[name])
    #             alls['指标变化率的时间序列'].append(col)
    #             alls['P值'].append(cotest[1])
    #             if cotest[1] > 0.05:
    #                 alls['是否协整'].append('否')
    #             else:
    #                 alls['是否协整'].append('是')
    #     else:
    #         if period == 'day':
    #             ys = y
    #             datanew = data
    #         if period == 'month':
    #             datanew = copy.deepcopy(data)
    #             datanew['year'] = datanew.index.year
    #             datanew['month'] = datanew.index.month
    #             datanew.set_index(['year', 'month'], drop=True, inplace=True)
    #             ys = copy.deepcopy(y)
    #             ys['year'] = ys.index.year
    #             ys['month'] = ys.index.month
    #             ys.drop_duplicates(['year', 'month'], keep = 'last', inplace = True)
    #             ys.set_index(['year', 'month'], drop = True, inplace = True)
    #         if period == 'quater':
    #             datanew = copy.deepcopy(data)
    #             datanew['year'] = datanew.index.year
    #             datanew['quater'] = quater(datanew.index)
    #             datanew.set_index(['year', 'quater'], drop = True, inplace = True)
    #             ys = copy.deepcopy(y)
    #             ys['year'] = ys.index.year
    #             ys['quater'] = quater(ys.index)
    #             ys.drop_duplicates(['year', 'quater'], keep = 'last', inplace = True)
    #             ys.set_index(['year', 'quater'], drop = True, inplace = True)
    #         for col in data.columns:
    #             target = ys.join(datanew[col], how = 'inner')
    #             target.dropna(inplace = True)
    #             cotest = coint(target[col], target[name])
    #             alls['指标变化率的时间序列'].append(col)
    #             alls['P值'].append(cotest[1])
    #             if cotest[1] > 0.05:
    #                 alls['是否协整'].append('否')
    #             else:
    #                 alls['是否协整'].append('是')
    #     alls = pd.DataFrame.from_dict(alls)
    #     alls = alls[['指标变化率的时间序列', 'P值', '是否协整']]
    #     return alls
