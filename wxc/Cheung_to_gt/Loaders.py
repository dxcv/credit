import time as t
import warnings
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
warnings.filterwarnings('ignore')
w.start()


class Loaders:
    def __init__(self):
        # 国债行情数据
        self.bond = ['S0059744', 'S0059745', 'S0059746', 'S0059747', 'S0059748', 'S0059749']
        self.bond_name = ['中债国债到期收益率:1年', '中债国债到期收益率:2年', '中债国债到期收益率:3年',
                          '中债国债到期收益率:5年', '中债国债到期收益率:7年', '中债国债到期收益率:10年']

        # 经济前景类指标
        self.prospect = ['M0017126', 'M0017127', 'M0017128', 'M0017129', 'M0017130', 'M0017131', 'M0017132',
                         'M0017133', 'M5766711', 'M0017134', 'M0017135', 'M0017136', 'M0017137', 'M5207790',
                         'M0000138', 'M0061603', 'M0290204']
        self.prospect_name = ['PMI', 'PMI:生产', 'PMI:新订单', 'PMI:新出口订单', 'PMI:在手订单', 'PMI:产成品库存',
                              'PMI:采购量', 'PMI:进口', 'PMI:出厂价格', 'PMI:主要原材料购进价格', 'PMI:原材料库存',
                              'PMI:从业人员', 'PMI:供货商配送时间', 'PMI:生产经营活动预期', '财新中国PMI',
                              '财新中国服务业PMI:经营活动指数', '财新中国综合PMI:产出指数']

        # 投资类指标
        self.invest = ['M0000315', 'M0000313', 'M0000323', 'M0000273', 'S0029657', 'M0000275', 'M0000357', 'M5440435',
                       'M5531328']
        self.invest_name = ['固定资产投资资金来源:国内贷款:累计同比', '固定资产投资资金来源:国家预算内资金:累计同比',
                            '固定资产投资资金来源:自筹资金:累计同比', '固定资产投资完成额:累计同比',
                            '房地产开发投资完成额:累计同比', '新增固定资产投资完成额:累计同比',
                            '固定资产投资完成额:制造业:累计同比', '固定资产投资完成额:基础设施建设投资:累计同比',
                            '固定资产投资完成额:基础设施建设投资(不含电力):累计同比']

        # 工业类指标
        self.industry = ['M0000011', 'M0096214', 'M0096215', 'M0068072', 'M0096216', 'M0000557', 'M5207464', 'M0000555',
                         'M0000561', 'M0044700']
        self.industry_name = ['工业增加值:累计同比', '工业增加值:采矿业:累计同比', '工业增加值:制造业:累计同比',
                              '工业增加值:汽车制造业:累计同比', '工业增加值:电力、燃气及水的生产和供应业:累计同比',
                              '工业企业:利润总额:累计同比', '工业企业:利润总额:当月同比', '工业企业:主营业务收入:累计同比',
                              '工业企业:产成品存货:累计同比', '工业企业:资产负债率']

        # 进出口类指标
        self.commerce = ['M0000607', 'M0000609', 'M0000611', 'M0000610', 'M0008498', 'M0007539', 'M0008652', 'M0054766',
                         'M0054774', 'M0054770']
        self.commerce_name = ['出口金额:当月同比', '进口金额:当月同比', '贸易差额:当月同比', '贸易差额:当月值',
                              '美国:进出口金额:累计同比', '日本:进出口金额:累计同比', '欧盟:进出口金额:累计同比',
                              '美国:贸易差额:当月值', '欧盟:贸易差额:当月值', '日本:贸易差额:当月值']

        # 房地产类指标
        self.house = ['S0029657', 'S0073284', 'S0073288', 'S0073285', 'S0073287', 'S0073290', 'S0073293', 'S0073297',
                      'S0073300', 'S0049591', 'S0029673', 'S0029672']
        self.house_name = ['房地产开发投资完成额:累计同比', '待开发土地面积:累计同比', '本年购置土地面积:累计同比',
                           '本年土地成交价款:累计同比', '土地购置费:累计同比', '房屋施工面积:累计同比',
                           '房屋新开工面积:累计同比', '房屋竣工面积:累计同比', '商品房销售面积:累计同比',
                           '商品房销售额:累计同比', '商品房待售面积:累计同比', '商品房待售面积:累计值']

        # 消费类指标
        self.consume = ['M0001440', 'M5405527', 'M0061665', 'M0061666', 'M0012303']
        self.consume_name = ['社会消费品零售总额:累计同比', '社会消费品零售总额:实际累计同比',
                             '社会消费品零售总额:城镇:累计同比', '社会消费品零售总额:乡村:累计同比', '消费者信心指数(月)']

        # 价格指数类指标
        self.index = ['M0000612', 'M0000616', 'M0000613', 'M0085932', 'M0096666', 'M0000614', 'M0000615', 'M0327903',
                      'M0044542', 'M0001227', 'M0001228', 'M0001229', 'M0001230', 'M0001231', 'M0001232', 'M0001233',
                      'M0001234', 'M0001235', 'M0001236']
        self.index_name = ['CPI:当月同比', 'CPI:食品:当月同比', 'CPI:非食品:当月同比',
                           'CPI:不包括食品和能源(核心CPI):当月同比', 'CPI:不包括鲜菜和鲜果:当月同比',
                           'CPI:消费品:当月同比', 'CPI:服务:当月同比', 'CPI:食品烟酒:当月同比',
                           'CPI:食品烟酒:畜肉类:猪肉:当月同比', 'PPI:全部工业品:当月同比', 'PPI:生产资料:当月同比',
                           'PPI:生产资料:采掘工业:当月同比', 'PPI:生产资料:原材料工业:当月同比',
                           'PPI:生产资料:加工工业:当月同比', 'PPI:生活资料:当月同比', 'PPI:生活资料:食品类:当月同比',
                           'PPI:生活资料:衣着类:当月同比', 'PPI:生活资料:一般日用品类:当月同比',
                           'PPI:生活资料:耐用消费品类:当月同比']

        # 发改委价格类指标(daily)
        self.ndrc = ['S5063746', 'S5063747', 'S5063748', 'S5063749', 'S5063750', 'S5063751', 'S5063752', 'S5063753',
                     'S5063754', 'S5063755', 'S5063756', 'S5063757', 'S5063758', 'S5063759', 'S5080521', 'S5080522',
                     'S5080523', 'S5080524', 'S5080525', 'S5080526', 'S5080527', 'S5080528', 'S5080529', 'S5080530',
                     'S5080531', 'S5080532', 'S5080533', 'S5080534', 'S5080535', 'S5080536']
        self.ndrc_name = ['36个城市平均零售价:晚籼米', '36个城市平均零售价:粳米', '36个城市平均零售价:面粉:富强粉',
                          '36个城市平均零售价:面粉:标准粉', '36个城市平均零售价:花生油:桶装',
                          '36个城市平均零售价:菜籽油:桶装', '36个城市平均零售价:豆油:桶装', '36个城市平均零售价:猪肉',
                          '36个城市平均零售价:牛肉', '36个城市平均零售价:羊肉', '36个城市平均零售价:鸡肉',
                          '36个城市平均零售价:鸡蛋', '36个城市平均零售价:草鱼', '36个城市平均零售价:鲢鱼',
                          '36个大中城市平均零售价:带鱼', '36个大中城市平均零售价:芹菜', '36个大中城市平均零售价:大白菜',
                          '36个大中城市平均零售价:萝卜', '36个大中城市平均零售价:土豆', '36个大中城市平均零售价:胡萝卜',
                          '36个大中城市平均零售价:尖椒', '36个大中城市平均零售价:圆白菜', '36个大中城市平均零售价:豆角',
                          '36个大中城市平均零售价:蒜苔', '36个大中城市平均零售价:韭菜', '36个大中城市平均零售价:青椒',
                          '36个大中城市平均零售价:黄瓜', '36个大中城市平均零售价:西红柿', '36个大中城市平均零售价:油菜',
                          '36个大中城市平均零售价:茄子']

        #  统计局生产资料类指标(every ten days)
        self.product = ['S5914455', 'S5914456', 'S5914457', 'S5914458', 'S5914459', 'S5914460', 'S5914461', 'S5914462',
                        'S5914463', 'S5914464', 'S5914465', 'S5914466', 'S5914467', 'S5914468', 'S5914469', 'S5914470',
                        'S5914471', 'S5914472', 'S5914473', 'S5914474', 'S5914475', 'S5914476', 'S5914477', 'S5914478',
                        'S5914479', 'S5914480', 'S5914481', 'S5914482', 'S5914483', 'S5914484', 'S5914485', 'S5914486',
                        'S5914487', 'S5914488', 'S5914489', 'S5914490', 'S5914491', 'S5914492', 'S5914493', 'S5914494',
                        'S5914495', 'S5914496', 'S5914497', 'S5914498', 'S5914499', 'S5914500', 'S5914501', 'S5914502',
                        'S5914503', 'S5914504']
        self.product_name = ['市场价:螺纹钢:HRB400 Φ16-25mm:全国', '市场价:线材:HPB300 Φ6.5mm:全国',
                             '市场价:普通中板:Q235 20mm:全国', '市场价:热轧普通薄板:Q235 3mm:全国',
                             '市场价:无缝钢管:20# 219*6:全国', '市场价:角钢:5#:全国', '市场价:电解铜:1#:全国',
                             '市场价:铝锭:A00:全国', '市场价:铅锭:1#:全国', '市场价:锌锭:0#:全国', '市场价:硫酸(98%):全国',
                             '市场价:液碱(32%):全国', '市场价:甲醇(优等品):全国', '市场价:石油苯(工业级):全国',
                             '市场价:苯乙烯(一级品):全国', '市场价:LLDPE(7042):全国', '市场价:聚丙烯(T30S):全国',
                             '市场价:聚氯乙烯(SG5):全国', '市场价:顺丁橡胶(BR9000):全国', '市场价:涤纶长丝(FDY150D/96F):全国',
                             '市场价:液化天然气LNG:全国', '市场价:液化气LPG:全国', '市场价:汽油(97#):全国',
                             '市场价:汽油(93#):全国', '市场价:柴油(0#):全国', '市场价:石蜡(58#半):全国',
                             '市场价:无烟煤(2号洗中块):全国', '市场价:普通混煤(Q4500):全国', '市场价:山西大混(Q5000):全国',
                             '市场价:山西优混(Q5500):全国', '市场价:大同混煤(Q5800):全国', '市场价:1/3焦煤:全国',
                             '市场价:二级冶金焦:全国', '市场价:复合硅酸盐水泥:P.C32.5 袋装:全国',
                             '市场价:普通硅酸盐水泥:P.O42.5 散装:全国', '市场价:浮法平板玻璃:4.8/5mm:全国',
                             '市场价:稻米:粳稻米:全国', '市场价:小麦:国标三等:全国', '市场价:玉米:黄玉米二等:全国',
                             '市场价:棉花:皮棉 白棉三级:全国', '市场价:生猪:外三元:全国', '市场价:大豆:黄豆:全国',
                             '市场价:豆粕:粗蛋白含量≥43%:全国', '市场价:花生:油料花生米:全国', '市场价:尿素(小颗料):全国',
                             '市场价:硫酸钾复合肥:全国', '市场价:草甘膦(95%原药):全国', '市场价:人造板:1220*2440*15mm:全国',
                             '市场价:纸浆:漂白化学浆:全国', '市场价:瓦楞纸:高强:全国']

        # 统计局食品类指标(every 10 days)
        self.food = ['S0109998', 'S0109999', 'S0109997', 'S0110000', 'S0110001', 'S0110002', 'S0110003', 'S0110004',
                     'S0110005', 'S0110006', 'S0110007', 'S0110008', 'S0110009', 'S0110010', 'S0110011', 'S0110012',
                     'S0110013', 'S0110014', 'S0110015', 'S0110016', 'S0110017', 'S0110018', 'S0110019', 'S0110020',
                     'S0110021', 'S0110022', 'S0110023']
        self.food_name = ['50个城市平均价:面粉:标准粉', '50个城市平均价:面粉:富强粉', '50个城市平均价:大米', 
                          '50个城市平均价:豆制品:豆腐', '50个城市平均价:花生油:压榨一级', '50个城市平均价:大豆油:5L桶装', 
                          '50个城市平均价:菜籽油:一级散装', '50个城市平均价:猪肉:猪肉后臀尖(后腿肉)', 
                          '50个城市平均价:猪肉:五花肉', '50个城市平均价:牛肉:腿肉', '50个城市平均价:羊肉:腿肉', 
                          '50个城市平均价:鸡:白条鸡', '50个城市平均价:鸡:鸡胸肉', '50个城市平均价:鸭', '50个城市平均价:鸡蛋', 
                          '50个城市平均价:活鲤鱼', '50个城市平均价:活草鱼', '50个城市平均价:带鱼', '50个城市平均价:大白菜', 
                          '50个城市平均价:油菜', '50个城市平均价:芹菜', '50个城市平均价:黄瓜', '50个城市平均价:西红柿',
                          '50个城市平均价:豆角', '50个城市平均价:土豆', '50个城市平均价:苹果:富士', '50个城市平均价:香蕉:国产']

        # 狭义流动性指标(daily)
        self.nliquidity = ['M0041652', 'M0041653', 'M0041654', 'M0041655', 'M0041656', 'M0041657', 'M0041658', 'M0041659',
                           'M0041660', 'M0041661', 'M0041662', 'M0220162', 'M0220163', 'M0220164', 'M0220165', 'M0220166',
                           'M0220167', 'M0220168', 'M0220169', 'M0220170', 'M0220171', 'M0220172', 'M0017138', 'M0017139',
                           'M0017140', 'M0017141', 'M0017142', 'M0017143', 'M0017144', 'M0017145', 'M1004511', 'M1004512',
                           'M1004513', 'M1004514', 'M1004515', 'M1004516', 'M1004517', 'M1004518', 'M1004519']
        self.nliquidity_name = ['银行间质押式回购加权利率:1天', '银行间质押式回购加权利率:7天',
                                '银行间质押式回购加权利率:14天', '银行间质押式回购加权利率:21天',
                                '银行间质押式回购加权利率:1个月', '银行间质押式回购加权利率:2个月',
                                '银行间质押式回购加权利率:3个月', '银行间质押式回购加权利率:4个月',
                                '银行间质押式回购加权利率:6个月', '银行间质押式回购加权利率:9个月',
                                '银行间质押式回购加权利率:1年', '存款类机构质押式回购加权利率:1天',
                                '存款类机构质押式回购加权利率:7天', '存款类机构质押式回购加权利率:14天',
                                '存款类机构质押式回购加权利率:21天', '存款类机构质押式回购加权利率:1个月',
                                '存款类机构质押式回购加权利率:2个月', '存款类机构质押式回购加权利率:3个月',
                                '存款类机构质押式回购加权利率:4个月', '存款类机构质押式回购加权利率:6个月',
                                '存款类机构质押式回购加权利率:9个月', '存款类机构质押式回购加权利率:1年',
                                'SHIBOR:隔夜', 'SHIBOR:1周', 'SHIBOR:2周', 'SHIBOR:1个月', 'SHIBOR:3个月', 'SHIBOR:6个月',
                                'SHIBOR:9个月', 'SHIBOR:1年', 'GC001:加权平均', 'GC002:加权平均', 'GC003:加权平均',
                                'GC004:加权平均', 'GC007:加权平均', 'GC014:加权平均', 'GC028:加权平均', 'GC091:加权平均',
                                'GC182:加权平均']

        # 广义流动性指标
        self.bliquidity = ['M0001383', 'M0001385', 'M5206730', 'M5201630', 'M5206731', 'M5201631', 'M0043417', 'M0043418',
                           'M0009969', 'M0009970', 'M0009940', 'M0009941', 'M0009974', 'M0048260', 'M0009975', 'M0009976',
                           'M0057874', 'M0057875', 'M0009977', 'M0057876', 'M0057877', 'M5540101']
        self.bliquidity_name = ['M1:同比', 'M2:同比', '社会融资规模:当月值', '社会融资规模:累计值',
                                '社会融资规模:新增人民币贷款:当月值', '社会融资规模:新增人民币贷款:累计值',
                                '金融机构:短期贷款余额', '金融机构:中长期贷款余额', '金融机构:各项贷款余额',
                                '金融机构:各项贷款余额:同比', '金融机构:各项存款余额', '金融机构:各项存款余额:同比',
                                '金融机构:新增人民币贷款:短期贷款及票据融资:当月值', '金融机构:新增人民币贷款:票据融资:当月值',
                                '金融机构:新增人民币贷款:中长期:当月值', '金融机构:新增人民币贷款:居民户:当月值',
                                '金融机构:新增人民币贷款:居民户:短期:当月值', '金融机构:新增人民币贷款:居民户:中长期:当月值',
                                '金融机构:新增人民币贷款:非金融性公司及其他部门:当月值', '金融机构:新增人民币贷款:非金融性公司:短期:当月值',
                                '金融机构:新增人民币贷款:非金融性公司:中长期:当月值', '金融机构:新增人民币贷款:非银行业金融机构']

        # 跨境资本流动类指标
        self.transboarder = ['M0010049', 'M0001681', 'M0327899']
        self.transboarder_name = ['官方储备资产:外汇储备', '货币当局:国外资产:外汇(中央银行外汇占款)',
                                  '金融机构:人民币:资金运用:中央银行外汇占款']

        # 汇率类指标
        self.rate = ['M0000271', 'G0002331', 'G0002334', 'G0002347', 'G0002329']
        self.rate_name = ['美元指数', '欧元兑美元', '美元兑日元', '英镑兑美元', '美元兑人民币元']

        # 内外利差类指标(还需做差)
        ## 1年
        self.diffrate1 = ['G0000886', 'G0008063', 'G0008146']
        self.diffrate1_name = ['美国-中国:国债收益率:1年', '德国-中国:国债收益率:1年', '法国-中国:国债收益率:1年']
        ## 2年
        self.diffrate2 = ['G0000887', 'G0008064', 'G0008147']
        self.diffrate2_name = ['美国-中国:国债收益率:2年', '德国-中国:国债收益率:2年', '法国-中国:国债收益率:2年']
        ## 3年
        self.diffrate3 = ['G0000888', 'G0008065']
        self.diffrate3_name = ['美国-中国:国债收益率:3年', '德国-中国:国债收益率:3年']
        ## 5年
        self.diffrate5 = ['G0000889', 'G0008066', 'G0008148', 'G0006352']
        self.diffrate5_name = ['美国-中国:国债收益率:5年', '德国-中国:国债收益率:5年', '法国-中国:国债收益率:5年',
                               '英国-中国:国债收益率:5年']
        ## 7年
        self.diffrate7 = ['G0000890', 'G0008067']
        self.diffrate7_name = ['美国-中国:国债收益率:7年', '德国-中国:国债收益率:7年']
        ## 10年
        self.diffrate10 = ['G0000891', 'G0008068', 'G1400003', 'G0006353']
        self.diffrate10_name = ['美国-中国:国债收益率:10年', '德国-中国:国债收益率:10年', '法国-中国:国债收益率:10年',
                                '英国-中国:国债收益率:10年']

        # 其他政府监管类指标（前三个需做差）
        self.supervise = ['M0017142', 'M0017153', 'M1001816', 'M5515275', 'M5515276', 'M5515277', 'M5515278', 'M0048631',
                          'M0012276', 'M0024271', 'M5207866', 'M5207874', 'M0024274', 'M0085818', 'M0148909', 'M0148920',
                          'M0041753', 'M0148907', 'M5639028', 'M5639035', 'M0001710']
        self.supervise_name = ['SHIBOR:3个月', '回购定盘利率:7天(FR007)', 'GC007:收盘价', '信托业:未来一年到期规模:合计',
                               '信托业:未来一年到期规模:集合信托', '信托业:未来一年到期规模:单一信托',
                               '信托业:未来一年到期规模:财产权信托', '保险公司:银行存款和债券:占资金运用余额比例',
                               '保险公司:保险资金运用余额:投资:债券投资', '基金管理公司管理资产规模:公募基金',
                               '公募基金份额:开放式基金:债券型', '公募基金净值:开放式基金:债券型',
                               '证券投资基金成交金额:当月值', '银行理财产品资金余额', '中债:债券托管量:国债:商业银行',
                               '中债:债券托管量:国债:保险机构', '待购回债券余额', '中债:债券托管量:国债:合计',
                               '债券市场发行债券:同业存单', '债券市场托管余额:同业存单', '其他存款性公司:对其他金融机构债权']

        # 货币政策类指标
        self.currency_rule = ['M0329541', 'M0329542', 'M5528822', 'M5528821', 'M0061614', 'M0061615', 'M0061616', 'M0062600',
                     'M0043821', 'M0061518', 'M0010096', 'M0010099', 'M0001685', 'M0001699']
        self.currency_rule_name = ['中期借贷便利(MLF):投放数量:6个月', '中期借贷便利(MLF):投放数量:1年', '抵押补充贷款(PSL):期末余额',
                          '抵押补充贷款(PSL):提供资金:当月新增', '公开市场操作:货币净投放', '公开市场操作:货币投放',
                          '公开市场操作:货币回笼', '逆回购:到期量', '人民币存款准备金率:中小型存款类金融机构(月)',
                          '人民币存款准备金率:大型存款类金融机构(月)', '超额存款准备金率(超储率):金融机构',
                          '超额存款准备金率(超储率):农村信用社', '货币当局:对其他存款性公司债权', '货币当局:政府存款']

        # 财政政策类指标
        self.fin_rule = ['M0041707', 'M0041710', 'M0057912', 'M0046167', 'M0024064', 'M0046169', 'M0024063']
        self.fin_rule_name = ['债券发行量:政府债券:累计值', '债券发行量:政策性银行债:累计值', '债券发行量:地方政府债:累计值',
                              '公共财政支出:累计同比', '公共财政支出:当月同比', '公共财政收入:累计同比', '公共财政收入:当月同比']

        # 行业监管类指标
        self.field = ['M0061670', 'M5201621', 'M5201622', 'M0061671', 'M5201620', 'M0002003', 'M0007451']
        self.field_name = ['商业银行:流动性比例', '商业银行:存贷比', '商业银行:人民币超额备付金率', '商业银行:资本充足率',
                           '商业银行:净息差', '不良贷款余额:商业银行', '商业银行:拨备覆盖率']

        # 资产策略类指标（频度不一样，故分开。第一类日频，第二类月频）
        ## daily
        ### year1
        self.strategy_day1 = ['M1004263', 'M1000213', 'S0059771', 'M0048392', 'M1004298', ]
        self.strategy_day1_name = ['中债国开债到期收益率:1年','中债商业银行普通债到期收益率(AAA):1年',
                                   '中债企业债到期收益率(AAA):1年', '中债铁道债到期收益率:1年',
                                   '中债地方政府债到期收益率(AAA):1年']
        ### year2
        self.strategy_day2 = ['M1004264', 'M1000212', 'S0059772', 'M0048393', 'M1004299']
        self.strategy_day2_name = ['中债国开债到期收益率:2年', '中债商业银行普通债到期收益率(AAA):2年',
                                   '中债企业债到期收益率(AAA):2年', '中债铁道债到期收益率:2年',
                                   '中债地方政府债到期收益率(AAA):2年']

        ### year3
        self.strategy_day3 = ['M1004265', 'M1000214', 'S0059773', 'M0048394', 'M1004300']
        self.strategy_day3_name = ['中债国开债到期收益率:3年', '中债商业银行普通债到期收益率(AAA):3年',
                                   '中债企业债到期收益率(AAA):3年', '中债铁道债到期收益率:3年',
                                   '中债地方政府债到期收益率(AAA):3年']

        ### year5
        self.strategy_day5 = ['M1004267', 'M1000216', 'S0059774', 'M0048395', 'M1004302']
        self.strategy_day5_name = ['中债国开债到期收益率:5年', '中债商业银行普通债到期收益率(AAA):5年',
                                   '中债企业债到期收益率(AAA):5年', '中债铁道债到期收益率:5年',
                                   '中债地方政府债到期收益率(AAA):5年']

        ### year7
        self.strategy_day7 = ['M1004269', 'M1000218', 'S0059775', 'M0048396', 'M1004304']
        self.strategy_day7_name = ['中债国开债到期收益率:7年', '中债商业银行普通债到期收益率(AAA):7年',
                                   '中债企业债到期收益率(AAA):7年', '中债铁道债到期收益率:7年',
                                   '中债地方政府债到期收益率(AAA):7年']

        ### year10
        self.strategy_day10 = ['M1004271', 'M1000220', 'S0059776', 'M0048397', 'M1004306']
        self.strategy_day10_name = ['中债国开债到期收益率:10年', '中债商业银行普通债到期收益率(AAA):10年',
                                    '中债企业债到期收益率(AAA):10年', '中债铁道债到期收益率:10年',
                                    '中债地方政府债到期收益率(AAA):10年']

        ### general
        self.strategy_day = ['M1006618', 'M1003521', 'M0096870']
        self.strategy_day_name = [ '同业存单到期收益率(AAA+):1年', '短期融资券到期收益率(AAA+):1年', '贷款基础利率(LPR):1年']

        ## monthly
        self.strategy_mon = ['M0001529', 'M0009969', 'M0251909', 'M0001548', 'M0062845', 'M0150191', 'M0001557']
        self.strategy_mon_name = ['金融机构:本外币:资金来源:各项存款:境内存款:住户存款:活期存款', '金融机构:各项贷款余额',
                                  '金融机构:本外币:资金运用:各项贷款:境内贷款:住户贷款:中长期贷款',
                                  '金融机构:本外币:资金运用:各项贷款:境内贷款:非金融企业及机关团体贷款',
                                  '金融机构:本外币:资金来源:各项存款:境内存款:非金融企业存款:活期存款',
                                  '金融机构:本外币:资金运用:债券投资', '金融机构:本外币:资金运用:总计']

    def get_bond(self):
        bond_code = ""
        now_date = t.strftime('%Y-%m-%d', t.localtime(t.time()))
        for i in range(len(self.bond)):
            bond_code += self.bond[i]
            if i < (len(self.bond) - 1):
                bond_code += ', '
        raw_data = w.edb(bond_code, "2010-01-01", now_date)
        data = pd.DataFrame()
        data['date'] = pd.Series(raw_data.Times)
        for i in range(len(raw_data.Codes)):
            data[self.bond_name[i]] = pd.Series(raw_data.Data[i])
        data.set_index(['date'], inplace=True)
        return data

    def get_prospect(self):
        """
        获取经济前景类指标
        :return:
        """
        prospect_code = ""
        now_date = t.strftime('%Y-%m-%d', t.localtime(t.time()))
        for i in range(len(self.prospect)):
            prospect_code += self.prospect[i]
            if i < (len(self.prospect) - 1):
                prospect_code += ', '
        raw_data = w.edb(prospect_code, "2010-01-01", now_date)
        data = pd.DataFrame()
        data['date'] = pd.Series(raw_data.Times)
        for i in range(len(raw_data.Codes)):
            data[self.prospect_name[i]] = pd.Series(raw_data.Data[i])
        data.set_index(['date'], inplace=True)
        return data

    def get_invest(self):
        """
        获取投资类指标
        :return:
        """
        invest_code = ""
        now_date = t.strftime('%Y-%m-%d', t.localtime(t.time()))
        for i in range(len(self.invest)):
            invest_code += self.invest[i]
            if i < (len(self.invest) - 1):
                invest_code += ', '
        raw_data = w.edb(invest_code, "2010-01-01", now_date)
        data = pd.DataFrame()
        data['date'] = pd.Series(raw_data.Times)
        for i in range(len(raw_data.Codes)):
            data[self.invest_name[i]] = pd.Series(raw_data.Data[i])
        data.set_index(['date'], inplace=True)
        return data

    def get_industry(self):
        """
        获取工业类指标
        :return:
        """
        industry_code = ""
        now_date = t.strftime('%Y-%m-%d', t.localtime(t.time()))
        for i in range(len(self.industry)):
            industry_code += self.industry[i]
            if i < (len(self.industry) - 1):
                industry_code += ', '
        raw_data = w.edb(industry_code, "2010-01-01", now_date)
        data = pd.DataFrame()
        data['date'] = pd.Series(raw_data.Times)
        for i in range(len(raw_data.Codes)):
            data[self.industry_name[i]] = pd.Series(raw_data.Data[i])
        data.set_index(['date'], inplace=True)
        return data

    def get_commerce(self):
        """
        获取进出口类指标
        :return:
        """
        commerce_code = ""
        now_date = t.strftime('%Y-%m-%d', t.localtime(t.time()))
        for i in range(len(self.commerce)):
            commerce_code += self.commerce[i]
            if i < (len(self.commerce) - 1):
                commerce_code += ', '
        raw_data = w.edb(commerce_code, "2010-01-01", now_date)
        data = pd.DataFrame()
        data['date'] = pd.Series(raw_data.Times)
        for i in range(len(raw_data.Codes)):
            data[self.commerce_name[i]] = pd.Series(raw_data.Data[i])
        data.set_index(['date'], inplace=True)
        return data

    def get_house(self):
        """
        获取房地产类指标
        :return:
        """
        house_code = ""
        now_date = t.strftime('%Y-%m-%d', t.localtime(t.time()))
        for i in range(len(self.house)):
            house_code += self.house[i]
            if i < (len(self.house) - 1):
                house_code += ', '
        raw_data = w.edb(house_code, "2010-01-01", now_date)
        data = pd.DataFrame()
        data['date'] = pd.Series(raw_data.Times)
        for i in range(len(raw_data.Codes)):
            data[self.house_name[i]] = pd.Series(raw_data.Data[i])
        data.set_index(['date'], inplace=True)
        return data

    def get_consume(self):
        """
        获取消费类指标
        :return:
        """
        consume_code = ""
        now_date = t.strftime('%Y-%m-%d', t.localtime(t.time()))
        for i in range(len(self.consume)):
            consume_code += self.consume[i]
            if i < (len(self.consume) - 1):
                consume_code += ', '
        raw_data = w.edb(consume_code, "2010-01-01", now_date)
        data = pd.DataFrame()
        data['date'] = pd.Series(raw_data.Times)
        for i in range(len(raw_data.Codes)):
            data[self.consume_name[i]] = pd.Series(raw_data.Data[i])
        data.set_index(['date'], inplace=True)
        return data

    def get_index(self):
        """
        获取价格指数类指标
        :return:
        """
        index_code = ""
        now_date = t.strftime('%Y-%m-%d', t.localtime(t.time()))
        for i in range(len(self.index)):
            index_code += self.index[i]
            if i < (len(self.index) - 1):
                index_code += ', '
        raw_data = w.edb(index_code, "2010-01-01", now_date)
        data = pd.DataFrame()
        data['date'] = pd.Series(raw_data.Times)
        for i in range(len(raw_data.Codes)):
            data[self.index_name[i]] = pd.Series(raw_data.Data[i])
        data.set_index(['date'], inplace=True)
        return data

    def get_ndrc(self):
        """
        获取发改委价格类指标
        :return:
        """
        ndrc_code = ""
        now_date = t.strftime('%Y-%m-%d', t.localtime(t.time()))
        for i in range(len(self.ndrc)):
            ndrc_code += self.ndrc[i]
            if i < (len(self.ndrc) - 1):
                ndrc_code += ', '
        raw_data = w.edb(ndrc_code, "2010-01-01", now_date)
        data = pd.DataFrame()
        data['date'] = pd.Series(raw_data.Times)
        for i in range(len(raw_data.Codes)):
            data[self.ndrc_name[i]] = pd.Series(raw_data.Data[i])
        data.set_index(['date'], inplace=True)
        return data

    def get_product(self):
        """
        获取统计局生产资料类指标
        :return:
        """
        product_code = ""
        now_date = t.strftime('%Y-%m-%d', t.localtime(t.time()))
        for i in range(len(self.product)):
            product_code += self.product[i]
            if i < (len(self.product) - 1):
                product_code += ', '
        raw_data = w.edb(product_code, "2010-01-01", now_date)
        data = pd.DataFrame()
        data['date'] = pd.Series(raw_data.Times)
        for i in range(len(raw_data.Codes)):
            data[self.product_name[i]] = pd.Series(raw_data.Data[i])
        data.set_index(['date'], inplace=True)
        return data

    def get_food(self):
        """
        获取统计局食品类指标
        :return:
        """
        food_code = ""
        now_date = t.strftime('%Y-%m-%d', t.localtime(t.time()))
        for i in range(len(self.food)):
            food_code += self.food[i]
            if i < (len(self.food) - 1):
                food_code += ', '
        raw_data = w.edb(food_code, "2010-01-01", now_date)
        data = pd.DataFrame()
        data['date'] = pd.Series(raw_data.Times)
        for i in range(len(raw_data.Codes)):
            data[self.food_name[i]] = pd.Series(raw_data.Data[i])
        data.set_index(['date'], inplace=True)
        return data

    def get_nliquidity(self):
        """
        获取狭义流动性指标
        :return:
        """
        nliquidity_code = ""
        now_date = t.strftime('%Y-%m-%d', t.localtime(t.time()))
        for i in range(len(self.nliquidity)):
            nliquidity_code += self.nliquidity[i]
            if i < (len(self.nliquidity) - 1):
                nliquidity_code += ', '
        raw_data = w.edb(nliquidity_code, "2010-01-01", now_date)
        data = pd.DataFrame()
        data['date'] = pd.Series(raw_data.Times)
        for i in range(len(raw_data.Codes)):
            data[self.nliquidity_name[i]] = pd.Series(raw_data.Data[i])
        data.set_index(['date'], inplace=True)
        return data

    def get_bliquidity(self):
        """
        获取广义流动性指标
        :return:
        """
        bliquidity_code = ""
        now_date = t.strftime('%Y-%m-%d', t.localtime(t.time()))
        for i in range(len(self.bliquidity)):
            bliquidity_code += self.bliquidity[i]
            if i < (len(self.bliquidity) - 1):
                bliquidity_code += ', '
        raw_data = w.edb(bliquidity_code, "2010-01-01", now_date)
        data = pd.DataFrame()
        data['date'] = pd.Series(raw_data.Times)
        for i in range(len(raw_data.Codes)):
            data[self.bliquidity_name[i]] = pd.Series(raw_data.Data[i])
        data.set_index(['date'], inplace=True)
        return data

    def get_transboarder(self):
        """
        获取跨境资本流动类指标
        :return:
        """
        transboarder_code = ""
        now_date = t.strftime('%Y-%m-%d', t.localtime(t.time()))
        for i in range(len(self.transboarder)):
            transboarder_code += self.transboarder[i]
            if i < (len(self.transboarder) - 1):
                transboarder_code += ', '
        raw_data = w.edb(transboarder_code, "2010-01-01", now_date)
        data = pd.DataFrame()
        data['date'] = pd.Series(raw_data.Times)
        for i in range(len(raw_data.Codes)):
            data[self.transboarder_name[i]] = pd.Series(raw_data.Data[i])
        data.set_index(['date'], inplace=True)
        return data

    def get_rate(self):
        """
        获取汇率类指标
        :return:
        """
        rate_code = ""
        now_date = t.strftime('%Y-%m-%d', t.localtime(t.time()))
        for i in range(len(self.rate)):
            rate_code += self.rate[i]
            if i < (len(self.rate) - 1):
                rate_code += ', '
        raw_data = w.edb(rate_code, "2010-01-01", now_date)
        data = pd.DataFrame()
        data['date'] = pd.Series(raw_data.Times)
        for i in range(len(raw_data.Codes)):
            data[self.rate_name[i]] = pd.Series(raw_data.Data[i])
        data.set_index(['date'], inplace=True)
        return data

    def get_diffrate(self, year):
        """
        获取内外利差类指标
        :return:
        """
        diffrate_code = ""
        now_date = t.strftime('%Y-%m-%d', t.localtime(t.time()))
        if year == 1:
            diffrate = self.diffrate1
            diffrate_name = self.diffrate1_name
        elif year == 2:
            diffrate = self.diffrate2
            diffrate_name = self.diffrate2_name
        elif year == 3:
            diffrate = self.diffrate3
            diffrate_name = self.diffrate3_name
        elif year == 5:
            diffrate = self.diffrate5
            diffrate_name = self.diffrate5_name
        elif year == 7:
            diffrate = self.diffrate7
            diffrate_name = self.diffrate7_name
        else:
            diffrate = self.diffrate10
            diffrate_name = self.diffrate10_name
        for i in range(len(diffrate)):
            diffrate_code += diffrate[i]
            if i < (len(diffrate) - 1):
                diffrate_code += ', '
        raw_data = w.edb(diffrate_code, "2010-01-01", now_date)
        data = pd.DataFrame()
        data['date'] = pd.Series(raw_data.Times)
        for i in range(len(raw_data.Codes)):
            bond = self.get_bond()['中债国债到期收益率:' + str(year) + '年']
            origin = pd.DataFrame({diffrate_name[i] : raw_data.Data[i],
                                   'date': pd.Series(raw_data.Times)}, index='date')
            bond.join(origin, how='inner')
            origin.set_index(['date'], inplace=True)

            data[diffrate_name[i]] = pd.Series(raw_data.Data[i])
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

    @staticmethod
    def quarter(data):
        """
        convert monthly to quarterly data
        :return:
        """
        q = (data.month // 3) + 1
        return q

    def cointest(self, data, period, bond):
        """
        协整检验
        :param data:
        :param period:
        :param bond: int, the type of bond
        :return:
        """
        alls = {'指标变化率的时间序列': [],
               'P值': [],
               '是否协整': []}
        name = '中债国债到期收益率:' + str(bond) + '年'
        y = self.get_bond()[[name]]
        if period == 'week':
            for col in data.columns:
                target = y.join(data[col], how='outer')
                target[name].fillna(inplace=True, method='ffill', limit=31)
                target.dropna(inplace=True)
                cotest = coint(target[col], target[name])
                alls['指标变化率的时间序列'].append(col)
                alls['P值'].append(cotest[1])
                if cotest[1] > 0.05:
                    alls['是否协整'].append('否')
                else:
                    alls['是否协整'].append('是')
        else:
            if period == 'day':
                ys = y
                datanew = data
            if period == 'month':
                datanew = copy.deepcopy(data)
                datanew['year'] = pd.Series([i.year for i in datanew.index], index=datanew.index)
                datanew['month'] = pd.Series([i.month for i in datanew.index], index=datanew.index)
                datanew.set_index(['year', 'month'], drop=True, inplace=True)
                ys = copy.deepcopy(y)
                ys['year'] = pd.Series([i.year for i in ys.index], index=ys.index)
                ys['month'] = pd.Series([i.month for i in ys.index], index=ys.index)
                ys.drop_duplicates(['year', 'month'], keep = 'last', inplace = True)
                ys.set_index(['year', 'month'], drop = True, inplace = True)
            if period == 'quarter':
                datanew = copy.deepcopy(data)
                datanew['year'] = pd.Series([i.year for i in datanew.index], index=datanew.index)
                datanew['quarter'] = pd.Series([self.quarter(i) for i in datanew.index], index=datanew.index)
                datanew.set_index(['year', 'quarter'], drop = True, inplace = True)
                ys = copy.deepcopy(y)
                ys['year'] = pd.Series([i.year for i in datanew.index], index=ys.index)
                ys['quarter'] = pd.Series([self.quarter(i) for i in datanew.index], index=ys.index)
                ys.drop_duplicates(['year', 'quarter'], keep = 'last', inplace = True)
                ys.set_index(['year', 'quarter'], drop = True, inplace = True)
            # print(ys)
            for col in data.columns:
                target = ys.join(datanew[col], how = 'inner')
                target.dropna(inplace = True)
                cotest = coint(target[col], target[name])
                alls['指标变化率的时间序列'].append(col)
                alls['P值'].append(cotest[1])
                if cotest[1] > 0.05:
                    alls['是否协整'].append('否')
                else:
                    alls['是否协整'].append('是')
        alls = pd.DataFrame.from_dict(alls)
        alls = alls[['指标变化率的时间序列', 'P值', '是否协整']]
        return alls

    def reg(self, data, period, bond):
        """
        一元回归检验
        :param data:
        :param period:
        :param bond:
        :return:
        """
        name = '中债国债到期收益率:' + str(bond) + '年'
        y = self.get_bond()[[name]]
        alls = {'指标变化率的时间序列': [],
                'coef': [],
                'Prob(F-statistic)': [],
                'R-squared': [],
                'No.Observations': []}
        if period == 'special day':
            for col in data.columns:
                target = y.join(data[col], how='outer')
                target[name].fillna(inplace=True, method='ffill', limit=31)
                target.dropna(inplace=True)
                model = sm.OLS(endog=target[name], exog=sm.add_constant(target[col]))
                res = model.fit()
                r2 = r2_score(target[name], res.fittedvalues)
                alls['指标变化率的时间序列'].append(col)
                alls['coef'].append(res.params[1])
                alls['Prob(F-statistic)'].append(res.pvalues[1])
                alls['R-squared'].append(r2)
                alls['No.Observations'].append(target.shape[0])
        else:
            if period == 'day':
                ys = y
                datanew = data
            if period == 'month':
                datanew = copy.deepcopy(data)
                datanew['year'] = pd.Series([i.year for i in datanew.index], index=datanew.index)
                datanew['month'] = pd.Series([i.month for i in datanew.index], index=datanew.index)
                datanew.set_index(['year', 'month'], drop=True, inplace=True)
                ys = copy.deepcopy(y)
                ys['year'] = pd.Series([i.year for i in ys.index], index=ys.index)
                ys['month'] = pd.Series([i.month for i in ys.index], index=ys.index)
                ys.drop_duplicates(['year', 'month'], keep='last', inplace=True)
                ys.set_index(['year', 'month'], drop=True, inplace=True)
            if period == 'quarter':
                datanew = copy.deepcopy(data)
                datanew['year'] = pd.Series([i.year for i in datanew.index], index=datanew.index)
                datanew['quarter'] = pd.Series([self.quarter(i) for i in datanew.index], index=datanew.index)
                datanew.set_index(['year', 'quarter'], drop=True, inplace=True)
                ys = copy.deepcopy(y)
                ys['year'] = pd.Series([i.year for i in datanew.index], index=ys.index)
                ys['quarter'] = pd.Series([self.quarter(i) for i in datanew.index], index=ys.index)
                ys.drop_duplicates(['year', 'quarter'], keep='last', inplace=True)
                ys.set_index(['year', 'quarter'], drop=True, inplace=True)
            for col in data.columns:
                target = ys.join(datanew[col], how='inner')
                target.dropna(inplace=True)
                model = sm.OLS(endog=target[name], exog=sm.add_constant(target[col]))
                res = model.fit()
                r2 = r2_score(target[name], res.fittedvalues)
                alls['指标变化率的时间序列'].append(col)
                alls['coef'].append(res.params[1])
                alls['Prob(F-statistic)'].append(res.pvalues[1])
                alls['R-squared'].append(r2)
                alls['No.Observations'].append(target.shape[0])
        alls = pd.DataFrame.from_dict(alls)
        alls = alls[['指标变化率的时间序列', 'coef', 'Prob(F-statistic)', 'R-squared', 'No.Observations']]
        return alls

    @staticmethod
    def __gen_results__(adf, coin, results):
        adf_ = adf.set_index('时间序列(差分处理)', drop=True)
        coin_ = coin.set_index('指标变化率的时间序列', drop=True)
        results_ = results.set_index('指标变化率的时间序列', drop=True)
        alls = results_[['coef', 'Prob(F-statistic)', 'R-squared']].join(adf_['是否平稳'], how='outer')
        alls = alls.join(coin_['是否协整'])
        return alls

    def adf_coin_reg(self, type, period, bond):
        """
        合并所有结果
        :param type: data type, eg: prospect
        :param period: frequency of type data
        :param bond: year of bond
        :return: combined selection result(not including elastic net)
        """
        if type == 'prospect':
            data = self.get_prospect()
        elif type == 'invest':
            data = self.get_invest()
        elif type == 'industry':
            data = self.get_industry()
        elif type == 'commerce':
            data = self.get_commerce()
        elif type == 'house':
            data = self.get_house()
        elif type == 'consume':
            data = self.get_consume()
        elif type == 'index':
            data = self.get_index()
        elif type == 'ndrc':
            data = self.get_ndrc()
        elif type == 'product':
            data = self.get_product()
        elif type == 'food':
            data = self.get_food()
        elif type == 'nliquidity':
            data = self.get_nliquidity()
        elif type == 'bliquidity':
            data = self.get_bliquidity()
        elif type == 'transboarder':
            data = self.get_transboarder()
        adf = self.ADFtest(data)
        coin = self.cointest(data, period, bond)
        regs = self.reg(data, period, bond)
        fin = self.__gen_results__(adf, coin, regs)
        return fin
